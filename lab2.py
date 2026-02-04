import cv2 as cv
import numpy as np

vid = cv.VideoCapture("bang_chuyen.mp4")


# how many frames to run red-line detection for before locking positions
DETECTION_LOCK_FRAMES = 30

trackers = []
next_id = 0
frame_count = 0

# smoothing and fade settings to reduce flicker
SMOOTH_ALPHA = 0.6
FADE_FRAMES = 4

# detection caps to avoid huge/merged-circle outlines
MAX_PEAK_RADIUS = 40
MAX_DETECTION_RADIUS = 80

# Detected vertical line x-positions (in pixels) and their counts
line_xs = []
line_counts = []

def detect_red_lines(frame):
    h, w = frame.shape[:2]
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower1 = np.array([0, 100, 100])
    upper1 = np.array([10, 255, 255])
    lower2 = np.array([160, 100, 100])
    upper2 = np.array([180, 255, 255])
    m1 = cv.inRange(hsv, lower1, upper1)
    m2 = cv.inRange(hsv, lower2, upper2)
    mask = cv.bitwise_or(m1, m2)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    xs = []
    for cnt in contours:
        x, y, wc, hc = cv.boundingRect(cnt)
        if hc > h * 0.4 and wc < w * 0.1:
            xs.append(x + wc // 2)
    xs = sorted(xs)
    return xs

def update_detected_lines(xs_new):
    global line_xs, line_counts
    if not xs_new:
        return
    if not line_xs:
        line_xs = xs_new.copy()
        line_counts = [0] * len(line_xs)
        return
    matched = [False] * len(xs_new)
    new_xs = []
    new_counts = []
    for i, oldx in enumerate(line_xs):
        best_j = None
        best_d = None
        for j, nx in enumerate(xs_new):
            if matched[j]:
                continue
            d = abs(oldx - nx)
            if best_d is None or d < best_d:
                best_d = d
                best_j = j
        if best_j is not None and best_d < 80:
            new_xs.append(xs_new[best_j])
            new_counts.append(line_counts[i])
            matched[best_j] = True
    for j, nx in enumerate(xs_new):
        if not matched[j]:
            new_xs.append(nx)
            new_counts.append(0)
    line_xs = new_xs
    line_counts = new_counts

def match_tracker(x, y, trackers, r=None, thresh=60):
    best_i = None
    best_d = None
    for i, t in enumerate(trackers):
        tx, ty = t['pos']
        tr = t.get('r', 20)
        d = np.hypot(tx - x, ty - y)
        # adaptive threshold based on sizes to avoid wrong large jumps
        adaptive = int(max(thresh, 0.5 * (tr + (r if r is not None else tr))))
        if best_d is None or d < best_d:
            best_d = d
            best_i = i
    if best_d is not None and best_d < adaptive:
        return best_i
    return None

while True:
    ret, frame = vid.read()
    if not ret:
        break
    if frame is None:
        continue

    h, w = frame.shape[:2]
    frame_count += 1
    # detect red vertical line(s) for the first few frames to stabilize, then lock
    if frame_count <= DETECTION_LOCK_FRAMES:
        xs_new = detect_red_lines(frame)
        if xs_new:
            update_detected_lines(xs_new)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    # tighten Hough parameters to reduce false positives and cap max radius
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1.2, minDist=40,
                              param1=50, param2=40, minRadius=6, maxRadius=MAX_DETECTION_RADIUS)

    # Additional contour-based detections (helps with occluded/merged/small circles)
    _, bw = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel3 = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    bw = cv.morphologyEx(bw, cv.MORPH_OPEN, kernel3, iterations=1)
    contours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    detections = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for c in circles[0, :]:
            detections.append((int(c[0]), int(c[1]), int(c[2])))

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < 50:
            continue
        perim = cv.arcLength(cnt, True)
        circ = 0
        if perim > 0:
            circ = 4 * np.pi * area / (perim * perim)
        (cx, cy), cr = cv.minEnclosingCircle(cnt)
        # Accept small circles or reasonably circular contours
        if circ > 0.4 or cr < 18:
            detections.append((int(cx), int(cy), int(cr)))
        else:
            # Large/merged contour: use distance transform to find multiple peaks
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv.drawContours(mask, [cnt], -1, 255, -1)
            dist = cv.distanceTransform(mask, cv.DIST_L2, 5)
            minVal, maxVal, minLoc, maxLoc = cv.minMaxLoc(dist)
            if maxVal > 6:
                # local maxima as peaks
                dil = cv.dilate(dist, np.ones((15, 15), np.uint8))
                peaks = (dist == dil) & (dist > 0.4 * maxVal)
                ys, xs = np.where(peaks)
                for yy, xx in zip(ys, xs):
                    peak_r = int(dist[yy, xx])
                    # ignore very large peaks (likely background/merged large area)
                    if peak_r <= MAX_PEAK_RADIUS:
                        detections.append((int(xx), int(yy), peak_r))

    # deduplicate detections (merge nearby ones)
    filtered = []
    for (x, y, r) in detections:
        keep = True
        for (fx, fy, fr) in filtered:
            if np.hypot(fx - x, fy - y) < max(8, 0.4 * min(fr, r)):
                keep = False
                break
        if keep:
            filtered.append((x, y, r))

    seen_ids = set()
    # feed detections into tracker matching
    for (x, y, r) in filtered:
        # cap detection radius to avoid sudden huge outlines
        if r > MAX_DETECTION_RADIUS:
            r = MAX_DETECTION_RADIUS
        i = match_tracker(x, y, trackers, r=r)
        if i is None:
            trackers.append({'id': next_id, 'pos': (x, y), 'r': r, 'smooth_pos': (x, y), 'smooth_r': r, 'counted_lines': set(), 'missed': 0, 'hits': 1})
            i = len(trackers) - 1
            next_id += 1
            prev_x = trackers[i]['pos'][0]
        else:
            prev_x = trackers[i]['pos'][0]

        # increase hit count for tracker (used to suppress spurious trackers)
        trackers[i]['hits'] = min(trackers[i].get('hits', 1) + 1, 10)

        # check crossing for each configured line (left->right)
        # only count if tracker has enough hits (seen on multiple frames)
        for li, lx in enumerate(line_xs):
            if trackers[i].get('hits', 0) >= 2:
                if (li not in trackers[i]['counted_lines']) and (prev_x < lx <= x):
                    line_counts[li] += 1
                    trackers[i]['counted_lines'].add(li)

        # update tracker with new observation (store radius too) and smooth position/radius
        old_sx, old_sy = trackers[i].get('smooth_pos', (x, y))
        sx = SMOOTH_ALPHA * x + (1 - SMOOTH_ALPHA) * old_sx
        sy = SMOOTH_ALPHA * y + (1 - SMOOTH_ALPHA) * old_sy
        trackers[i]['smooth_pos'] = (sx, sy)
        trackers[i]['smooth_r'] = SMOOTH_ALPHA * r + (1 - SMOOTH_ALPHA) * trackers[i].get('smooth_r', r)
        trackers[i]['pos'] = (x, y)
        trackers[i]['r'] = r
        trackers[i]['missed'] = 0
        seen_ids.add(trackers[i]['id'])

    # increment missed counters and remove very stale or low-confidence trackers
    for t in trackers:
        if t['id'] not in seen_ids:
            t['missed'] += 1

    new_trackers = []
    for t in trackers:
        # drop trackers that were only seen once and then missed for a while (likely false)
        if t['missed'] > 10 and t.get('hits', 0) < 2:
            continue
        # drop trackers that are extremely stale
        if t['missed'] > 60:
            continue
        new_trackers.append(t)
    trackers = new_trackers

    # draw UI: per-line counts
    font = cv.FONT_HERSHEY_SIMPLEX

    # draw labels and counts only (do not draw extra vertical lines)
    for idx, lx in enumerate(line_xs):
        label = f"LINE {idx+1}"
        cv.putText(frame, label, (lx + 8, 30), font, 0.7, (0, 0, 255), 2, cv.LINE_AA)
        cv.putText(frame, f"Count: {line_counts[idx]}", (lx + 8, 60), font, 0.6, (0, 255, 0), 2, cv.LINE_AA)

    # draw red 1-pixel outline with smoothing and short fade to avoid flicker
    for t in trackers:
        if t.get('hits', 0) < 2:
            continue
        missed = t.get('missed', 0)
        if missed > FADE_FRAMES:
            continue
        sx, sy = t.get('smooth_pos', t['pos'])
        sr = int(t.get('smooth_r', t.get('r', 16)))
        alpha = 1.0 if missed == 0 else max(0.15, 1.0 - float(missed) / FADE_FRAMES)
        tmp = frame.copy()
        cv.circle(tmp, (int(sx), int(sy)), sr, (0, 0, 255), 1)
        cv.addWeighted(tmp, alpha, frame, 1.0 - alpha, 0, frame)

    cv.imshow("video", frame)
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

vid.release()
cv.destroyAllWindows()