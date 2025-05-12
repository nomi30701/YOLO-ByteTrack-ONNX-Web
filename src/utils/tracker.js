import * as math from "mathjs";
import { minWeightAssign } from "munkres-algorithm";

const BYTE_TRACKER_CONFIG = {
  track_high_thresh: 0.25, // threshold for the first association (default 0.25)
  track_low_thresh: 0.1, // threshold for the second association
  new_track_thresh: 0.25, // threshold for init new track if the detection does not match any tracks
  track_buffer: 60, // buffer to calculate the time when to remove tracks (default 30)
  match_thresh: 0.8, // threshold for matching tracks (default 0.8)
  fuse_score: true, // Whether to fuse scoreidence scores with the iou distances before matching
};

// manange track status
class STrack {
  constructor(track_id, xywh, score, cls_idx, kalman_filter) {
    this.track_id = track_id;
    this.xywh = xywh; // [x_center, y_center, width, height]
    this.score = score;
    this.cls_idx = cls_idx;
    this.kalman_filter = kalman_filter;
    this.state = "Tracked"; // Tracked, Lost, Removed
    [this.mean, this.covariance] = kalman_filter.initiate(xywh); // init filter status
    this.is_activated = false;
    this.frame_id = 0;
  }

  xywhToTlwh(xywh) {
    return [xywh[0] - xywh[2] / 2, xywh[1] - xywh[3] / 2, xywh[2], xywh[3]];
  }

  // apply kalman filter for predict object next status (mean, convariance).
  // mean: [x, y, w, h, vx, vy, vw, vh]
  predict() {
    [this.mean, this.covariance] = this.kalman_filter.predict(
      this.mean,
      this.covariance
    );
    // update xywh
    this.xywh = [
      this.mean.get([0]),
      this.mean.get([1]),
      this.mean.get([2]),
      this.mean.get([3]),
    ];
  }

  // update track status (next frame) with new detection
  update(new_track, frame_id) {
    this.xywh = new_track.xywh.slice();
    this.score = new_track.score;
    this.cls_idx = new_track.cls_idx;
    [this.mean, this.covariance] = this.kalman_filter.update(
      this.mean,
      this.covariance,
      this.xywh
    );
    this.frame_id = frame_id;
    this.state = "Tracked";
    this.is_activated = true;
  }

  markLost() {
    this.state = "Lost";
  }

  markRemoved() {
    this.state = "Removed";
  }
}

/**
 * STrackPool class.
 * Manages the pool of STrack objects to reduce memory allocation overhead.
 */
class STrackPool {
  constructor(maxPoolSize = 100) {
    this.pool = [];
    this.maxPoolSize = maxPoolSize;
  }

  acquire(track_id, xywh, score, cls_idx, kalman_filter) {
    if (this.pool.length > 0) {
      const track = this.pool.pop();
      track.track_id = track_id;
      track.xywh = xywh.slice();
      track.score = score;
      track.cls_idx = cls_idx;
      track.kalman_filter = kalman_filter;
      track.state = "Tracked";
      track.is_activated = false;
      track.frame_id = 0;
      [track.mean, track.covariance] = kalman_filter.initiate(xywh);
      return track;
    }
    return new STrack(track_id, xywh, score, cls_idx, kalman_filter);
  }

  release(track) {
    if (this.pool.length < this.maxPoolSize) {
      this.pool.push(track);
    }
  }
}

/**
 * ByteTracker class.
 *
 * Processes the tracking of objects in a video stream.
 *
 * Update fnc: [{xywh: [x_center, y_center, width, height], score, cls_idx}, ...]
 */
export class BYTETracker {
  constructor() {
    this.tracked_STracks = [];
    this.lost_STracks = [];
    this.removed_STracks = [];
    this.frame_id = 0;
    this.kalman_filter = new KalmanFilterXYWH();
    this.track_id_count = 0;

    // add STrackPool to manage track objects
    this.trackPool = new STrackPool(200);
    this.arrayPool = [];
  }

  getTempArray(size) {
    if (this.arrayPool.length > 0) {
      const arr = this.arrayPool.pop();
      arr.length = size;
      return arr;
    }
    return new Array(size);
  }

  releaseArray(arr) {
    if (this.arrayPool.length < 20) {
      arr.length = 0; // clear the array
      this.arrayPool.push(arr);
    }
  }

  update(detections) {
    this.frame_id++;

    // New detections to STrack class
    let dets = this.getTempArray(detections.length);
    for (let i = 0; i < detections.length; i++) {
      dets[i] = this.trackPool.acquire(
        -1,
        detections[i].xywh,
        detections[i].score,
        detections[i].cls_idx,
        this.kalman_filter
      );
    }

    // Predict the next status (mean, covariance) of current tracked and lost tracks
    this.tracked_STracks.forEach((track) => track.predict());
    this.lost_STracks.forEach((track) => track.predict());

    // First association
    // filter high score dets and match with tracked_STracks
    let high_score_dets = dets.filter(
      (det) => det.score >= BYTE_TRACKER_CONFIG.track_high_thresh
    );

    let [matched1, _, unmatched_dets1] = this.match(
      this.tracked_STracks,
      high_score_dets
    ); // return index

    // Update matched tracks with high score dets
    let keepTrackFlags = new Array(this.tracked_STracks.length).fill(false);
    matched1.forEach(([track_idx, det_idx]) => {
      this.tracked_STracks[track_idx].update(
        high_score_dets[det_idx],
        this.frame_id
      );
      keepTrackFlags[track_idx] = true;
    });

    // Process unmatched tracks
    for (let i = 0; i < this.tracked_STracks.length; i++) {
      if (keepTrackFlags[i]) continue;
      this.tracked_STracks[i].markLost();
      this.lost_STracks.push(this.tracked_STracks[i]);
    }

    // Efficiently reorganize the tracked_STracks array (no filter)
    let j = 0;
    for (let i = 0; i < this.tracked_STracks.length; i++) {
      if (keepTrackFlags[i]) {
        if (i !== j) this.tracked_STracks[j] = this.tracked_STracks[i];
        j++;
      }
    }
    this.tracked_STracks.length = j;

    // Second association (low score det and lost tracks)
    let remaining_dets = unmatched_dets1
      .map((idx) => high_score_dets[idx])
      .concat(
        dets.filter((det) => det.score < BYTE_TRACKER_CONFIG.track_high_thresh)
      ); // get unmatch "high_score_dets" by "unmatched_dets1" index and concat with "low score dets"
    let [matched2, unmatched_tracks2, unmatched_dets2] = this.match(
      this.lost_STracks,
      remaining_dets
    ); // match "lost_STracks" with "remaining_dets"

    // Update matched lost tracks
    matched2.forEach(([track_idx, det_idx]) => {
      this.lost_STracks[track_idx].update(
        remaining_dets[det_idx],
        this.frame_id
      ); // matched lost tracks update status
      this.tracked_STracks.push(this.lost_STracks[track_idx]);
    }); // push to tracked_STracks
    this.lost_STracks = this.lost_STracks.filter(
      (_, idx) => !matched2.some(([t_idx]) => t_idx === idx)
    ); // remove matched lost tracks from lost_STracks

    // Init new tracks (unmatched detections)
    unmatched_dets2.forEach((det_idx) => {
      let track = remaining_dets[det_idx]; // get new track
      if (track.score >= BYTE_TRACKER_CONFIG.new_track_thresh) {
        track.track_id = this.track_id_count++;
        track.frame_id = this.frame_id;
        this.tracked_STracks.push(track);
      }
    });

    // remove over 30 frames lost tracks
    this.lost_STracks = this.lost_STracks.filter(
      (track) =>
        this.frame_id - track.frame_id <= BYTE_TRACKER_CONFIG.track_buffer
    );

    // release useless STrack objects pool
    for (let i = 0; i < dets.length; i++) {
      // if dets is not matched to any track, release it back to pool
      if (dets[i].track_id === -1) {
        this.trackPool.release(dets[i]);
      }
    }
    this.releaseArray(dets);

    const result = new Array(this.tracked_STracks.length);
    for (let i = 0; i < this.tracked_STracks.length; i++) {
      const track = this.tracked_STracks[i];
      result[i] = {
        track_id: track.track_id,
        tlwh: track.xywhToTlwh(track.xywh),
        score: track.score,
        cls_idx: track.cls_idx,
        state: track.state,
      };
    }

    return result;
  }

  match(tracks, detections) {
    if (tracks.length === 0) {
      return [[], [], Array.from({ length: detections.length }, (_, i) => i)];
    }
    if (detections.length === 0) {
      return [[], Array.from({ length: tracks.length }, (_, i) => i), []];
    }

    let cost_matrix = this.getDists(tracks, detections);
    let result = minWeightAssign(cost_matrix);
    let assignments = result.assignments;

    let matched = [];
    let unmatched_tracks = [];
    let assigned_dets = new Set();

    // Process the assignments
    for (let t_idx = 0; t_idx < assignments.length; t_idx++) {
      let d_idx = assignments[t_idx];
      if (
        d_idx !== null &&
        cost_matrix[t_idx][d_idx] <= BYTE_TRACKER_CONFIG.match_thresh
      ) {
        matched.push([t_idx, d_idx]);
        assigned_dets.add(d_idx);
      } else {
        unmatched_tracks.push(t_idx);
      }
    }

    // get unmatched dets
    let unmatched_dets = [];
    for (let d_idx = 0; d_idx < detections.length; d_idx++) {
      if (!assigned_dets.has(d_idx)) {
        unmatched_dets.push(d_idx);
      }
    }

    return [matched, unmatched_tracks, unmatched_dets];
  }
  getDists(tracks, detections) {
    const tracks_tlwh = tracks.map((t) => t.xywhToTlwh(t.xywh));
    const dets_tlwh = detections.map((d) => d.xywhToTlwh(d.xywh));
    const dists = new Array(tracks.length);
    // const DIST_THRESH = Math.max(...tracks_tlwh.map((t) => t[2] + t[3])) * 2;
    for (let i = 0; i < tracks.length; i++) {
      const row = new Array(detections.length);
      const [tx, ty] = tracks_tlwh[i];
      const DIST_THRESH = (tracks_tlwh[i][2] + tracks_tlwh[i][3]) * 1.5;
      for (let j = 0; j < detections.length; j++) {
        const [dx, dy] = dets_tlwh[j];
        if (Math.abs(tx - dx) + Math.abs(ty - dy) > DIST_THRESH) {
          row[j] = 1;
          continue;
        }
        row[j] = 1 - computeIoU(tracks_tlwh[i], dets_tlwh[j]);
      }
      dists[i] = row;
    }
    return dists;
  }
}

/**
 *
 * @param {Array[number]} box1 - [tlx, tly, w, h]
 * @param {Array[number]} box2 - [tlx, tly, w, h]
 * @returns {number} - IoU
 */
function computeIoU(box1, box2) {
  let [x1, y1, w1, h1] = box1; // tlwh
  let [x2, y2, w2, h2] = box2;
  let x_left = Math.max(x1, x2);
  let y_top = Math.max(y1, y2);
  let x_right = Math.min(x1 + w1, x2 + w2);
  let y_bottom = Math.min(y1 + h1, y2 + h2);
  if (x_right <= x_left || y_bottom <= y_top) return 0;
  let intersection = (x_right - x_left) * (y_bottom - y_top);
  let union = w1 * h1 + w2 * h2 - intersection;
  return intersection / union;
}

class KalmanFilterXYWH {
  constructor() {
    // State transition matrix F
    this.F = math.matrix([
      [1, 0, 0, 0, 1, 0, 0, 0],
      [0, 1, 0, 0, 0, 1, 0, 0],
      [0, 0, 1, 0, 0, 0, 1, 0],
      [0, 0, 0, 1, 0, 0, 0, 1],
      [0, 0, 0, 0, 1, 0, 0, 0],
      [0, 0, 0, 0, 0, 1, 0, 0],
      [0, 0, 0, 0, 0, 0, 1, 0],
      [0, 0, 0, 0, 0, 0, 0, 1],
    ]);
    // "H" is the observation matrix. It maps the state space to the measurement space.
    this.H = math.matrix([
      [1, 0, 0, 0, 0, 0, 0, 0],
      [0, 1, 0, 0, 0, 0, 0, 0],
      [0, 0, 1, 0, 0, 0, 0, 0],
      [0, 0, 0, 1, 0, 0, 0, 0],
    ]);
  }

  initiate(measurement) {
    const mean = math.matrix([...measurement, 0, 0, 0, 0]); // [x, y, w, h, vx, vy, vw, vh]
    const std = 2 * 0.05 * measurement[3]; // std_weight_position = 1/20, 2 times for velocity
    const P = math.diag([
      std * std,
      std * std,
      std * std,
      std * std,
      1e-2,
      1e-2,
      1e-2,
      1e-2,
    ]);
    return [mean, P];
  }

  // according to the state: "mean" "convarience", motion model: "transition matrix F" "covariance matrix P"
  // to predict the next state and covariance
  predict(mean, covariance) {
    // Q is the process noise covariance matrix. It represents the uncertainty in the state transition.
    const std_pos = 0.05 * mean.get([3]); // std_weight_position = 1/20
    const std_vel = 0.5 * mean.get([3]); // std_weight_velocity = 1/2
    const Q = math.diag([
      std_pos * std_pos,
      std_pos * std_pos,
      std_pos * std_pos,
      std_pos * std_pos,
      std_vel * std_vel,
      std_vel * std_vel,
      std_vel * std_vel,
      std_vel * std_vel,
    ]);
    const motion_mean = math.multiply(this.F, mean);
    const motion_cov = math.add(
      math.multiply(math.multiply(this.F, covariance), math.transpose(this.F)),
      Q
    );
    return [motion_mean, motion_cov];
  }

  update(mean, covariance, measurement) {
    // R is the measurement noise covariance matrix. It represents the uncertainty in the measurements.
    const std_pos = 0.05 * mean.get([3]); // std_weight_position = 1/20
    const std_w = 1e-1;
    const R = math.diag([
      std_pos * std_pos, // x
      std_pos * std_pos, // y
      std_w * std_w, // w
      std_pos * std_pos, // h
    ]);
    const z = math.matrix(measurement);
    const H_covar = math.multiply(this.H, covariance);
    const projected_mean = math.multiply(this.H, mean);
    const projected_cov = math.add(
      math.multiply(H_covar, math.transpose(this.H)),
      R
    );
    const K = math.multiply(
      covariance,
      math.transpose(this.H),
      math.inv(projected_cov)
    );
    const innovation = math.subtract(z, projected_mean);
    const updated_mean = math.add(mean, math.multiply(K, innovation));
    const updated_cov = math.subtract(covariance, math.multiply(K, H_covar));
    return [updated_mean, updated_cov];
  }
}
