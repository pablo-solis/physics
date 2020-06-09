import tensorflow as tf
import cv2
import time
import argparse

import posenet

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=100) # 100 seems to be the sweet spot
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.5)  #what does this do?
#parser.add_argument('--file', type=str, default=None, help="Optionally use a video file instead of a live camera")
parser.add_argument('--draw', type=int, default=0) # set to 1 to see skeleton being scored
parser.add_argument('--draw_angles', type=int, default=0) # set to 1 to see angles
parser.add_argument('--min_pose_score', type=float, default=0)
parser.add_argument('--angle_limit', type=int, default=3)
parser.add_argument('--capture_rate',type=int, default=1)
parser.add_argument('--score_angles',type=int,default=0)
args = parser.parse_args()

# CONSTANTS
SECTIONS = {'LEGS':["leftKnee", "rightKnee", "leftAnkle", "rightAnkle"],
            'ARMS':[]}

def run_posenet_from_sess(sess,model_outputs,input_image,
    scale_factor=args.scale_factor,
    output_stride=16,
    num_pose_detections = 3,
    min_score_in_multi_decode = args.min_pose_score, output_scale=1.0):

    # input needed from this point forward is:
    # input_image

    # intermediate output from a capture
    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(model_outputs, feed_dict={'image:0': input_image})

    # final output from a capture
    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multi.decode_multiple_poses(
        heatmaps_result.squeeze(axis=0),
        offsets_result.squeeze(axis=0),
        displacement_fwd_result.squeeze(axis=0),
        displacement_bwd_result.squeeze(axis=0),
        output_stride=output_stride,
        max_pose_detections=num_pose_detections,
        min_pose_score=min_score_in_multi_decode)

    keypoint_coords *= output_scale
    return pose_scores, keypoint_scores, keypoint_coords

def main():

    with tf.Session() as sess:
        # initialization
        model_cfg, model_outputs = posenet.load_model(args.model, sess)
        output_stride = model_cfg['output_stride']

        # Video
        # source
        #if args.file is not None:
        #    cap = cv2.VideoCapture(args.file)
        #else:
        cap = cv2.VideoCapture(args.cam_id)
        cap.set(3, args.cam_width)
        cap.set(4, args.cam_height)

        start = time.time()
        frame_count = 0

        # create pose slice
        slice = posenet.pose_slice()
        while True:
            # always do step 1
            # step 1
            input_image, display_image, output_scale = posenet.read_cap(cap,
            scale_factor=args.scale_factor,
            output_stride=output_stride)

            overlay_image = display_image # when main if is not run
            # main if
            if frame_count % args.capture_rate == 0:
                # run posenet

                # dispaly image is part of step 1 of running posenet
                pose_scores, kp_scores,kp_coords = run_posenet_from_sess(sess,model_outputs, input_image,
                    scale_factor=args.scale_factor,
                    output_stride=output_stride,
                    num_pose_detections = 3,
                    min_score_in_multi_decode = args.min_pose_score,output_scale=output_scale)
                # push into stack
                slice.push_pose(kp_coords, kp_scores, pose_scores)
                # stack.push(coords)

                if args.score_angles ==1:

                    pass

                # draw angles
                if args.draw_angles ==1:
                    overlay_image = posenet.draw_angles(display_image,pose_scores,kp_coords,limit = args.angle_limit)

                # overlay_image
                if args.draw == 1:
                    overlay_image = posenet.draw_skel_and_kp(
                        display_image,
                        pose_scores,
                        kp_scores,
                        kp_coords,
                        min_pose_score=0.15,
                        min_part_score=0.1)
            cv2.imshow('posenet', overlay_image)
            frame_count += 1
            accel = slice.avg_accel()
            vel = slice.avg_velocity()
            score = slice.score_function(func_input = accel/50, noise_input = vel, noise = 0.6)
            score = 0 if slice.is_chillin() else score
            print('score: ',  score)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print('Average FPS: ', frame_count / (time.time() - start))

if __name__ == "__main__":
    main()
