import cv2
import numpy as np
import posenet.constants
from posenet.utils import *

class pose_slice:
    def __init__(self,pose=None, confidence_scores = None, scalar_scores = None,fps = 3,lagNum = 3):
        self.lagNum = lagNum
        self.pose = [pose]*self.lagNum
        self.scalar_scores = [scalar_scores]*self.lagNum
        self.confidence_scores = [confidence_scores]*self.lagNum
        self.joint_names = ['leftShoulder','leftElbow', 'leftHip','leftKnee','rightShoulder','rightElbow', 'rightHip','rightKnee']
        self.joint_angle_velocities = {i:0 for i in self.joint_names}
        self.joint_angle_accel = {i:0 for i in self.joint_names}
        self.fps = fps
        if not pose:
            self.isEmpty = True
        else:
            self.isEmpty = False
    def push_pose(self,coords,scores, pose_score):
        if self.isEmpty == True:
            self.pose = [coords]*self.lagNum
            self.scores = [scores]*self.lagNum
            self.scalar_scores = [pose_score]*self.lagNum
            pose_angles = self.calculate_angles_from_pose(-1) # use last element of self.pose
            self.angles = [pose_angles]*self.lagNum
            self.isEmpty = False
        else:
            self.pose = [self.pose[i] for i in range(1,self.lagNum)]+[coords]
            self.confidence_scores = [self.confidence_scores[i] for i in range(1,self.lagNum)]+[scores]
            self.scalar_scores = [self.scalar_scores[i] for i in range(1,self.lagNum)]+[pose_score]
            pose_angles = self.calculate_angles_from_pose(-1)
            self.angles = [self.angles[i] for i in range(1,self.lagNum)]+[pose_angles]
            self.update_angle_accel()
            self.update_angle_velocities()
    def update_angle_velocities(self):
        if self.isEmpty:
            pass
        else:
            # for each joint calculate |angle_{i} - angle_{i-1}|
            vel_joint_values = {joint:
                [abs(self.angles[i][joint] - self.angles[i-1][joint]) for i in range(1,len(self.angles))]
                for joint in self.joint_names}
            for i in self.joint_names:
                # update each joint with the new avg
                self.joint_angle_velocities[i] = sum(vel_joint_values[i])/len(vel_joint_values[i])
    def update_angle_accel(self):
        if self.isEmpty:
            pass
        else:
            # for each joint calculate |v_{i} - 2*v_{i-1}+v_{i-2}|
            accel_joint_values = {joint:
                [abs(self.angles[i][joint] - 2*self.angles[i-1][joint]+self.angles[i-2][joint]) for i in range(2,len(self.angles))]
                for joint in self.joint_names}
            for i in self.joint_names:
                # update each joint with the new avg
                self.joint_angle_accel[i] = sum(accel_joint_values[i])/len(accel_joint_values[i])

    def avg_velocity(self):
        return sum(self.joint_angle_velocities.values())/len(self.joint_angle_velocities)
    def avg_accel(self):
        return sum(self.joint_angle_accel.values())/len(self.joint_angle_accel)
    def score_function(self,func_input, noise_input,noise = 0.3):
        # make more elaborate later (hyperbolic tangent)
        return np.tanh(func_input)/self.fps if np.tanh(noise_input)>noise else 0
    def get_single_angle(self,v1,v2):
        '''cos(angle) = v1@v2 /|v1||v2|'''
        # normalize
        v1 = v1/np.linalg.norm(v1)
        v2 = v2/np.linalg.norm(v2)
        return round(np.arccos(v1@v2)*180/np.pi,0) # units of degrees
    def calculate_angles_from_pose(self,idx):
        '''idx is the index to use for self.pose'''
        pose = self.pose[idx]
        score = self.scalar_scores[idx]
        ans = {}

        # just calculate score from most confident pose
        i = list(score).index(max(score))
        for joint,ends in posenet.JOINT_MAP_INDEX.items():
            end1 = ends[0]
            end2 = ends[1]
            # vector points from joint to end
            v1 = pose[i,end1,:] - pose[i,joint,:]
            v2 = pose[i,end2,:] - pose[i,joint,:]

            angle = self.get_single_angle(v1,v2)
            ans[posenet.PART_NAMES[joint]] = angle
        return ans
    def is_chillin(self):
        angles = self.angles[-1] # get last frame..
        leftShoulder = 0<= angles['leftShoulder']<=60
        rightShoulder = 0<= angles['rightShoulder']<=60
        shoulders = leftShoulder and rightShoulder
        hips = angles['leftHip']>150 and angles['rightHip']>150
        knees = angles['leftKnee']>120 and angles['rightKnee']>120
        # TODO check for acceleration and velocity
        if shoulders and hips and knees:
            return True
        else:
            return False





class pose_lag:
    def __init__(self,pose = None,
            scores = None,
            width = 1280,
            height = 720,
            fps = 3, lagNum=3):
        self.lagNum = lagNum
        self.pose = [pose]*self.lagNum
        self.scores = [scores]*self.lagNum
        self.torso = [5,6,11,12]
        self.arms = [7,8,9,10]
        self.legs = [13,14,15,16]
        self.arms_score = {'vel':0, 'accel':0, 'jerk':0}
        self.torso_score = {'vel':0, 'accel':0, 'jerk':0}
        self.legs_score = {'vel':0, 'accel':0, 'jerk':0}
        self.width = width
        self.height = height
        self.fps = fps # empirically about 4

        if not pose:
            self.isEmpty = True
        else:
            self.isEmpty = False


    def push_pose(self,coords,scores):
        if self.isEmpty == True:
            self.pose = [coords]*self.lagNum
            self.scores = [scores]*self.lagNum
            self.isEmpty = False
        else:
            self.pose = [self.pose[i] for i in range(1,self.lagNum)]+[coords]
            self.scores = [self.scores[i] for i in range(1,self.lagNum)]+[scores]
        # update all the things when a push happens
        self.normalize_pose_frame()
        self.update_body_scores(v_a_j = 'accel')
        self.update_body_scores(v_a_j = 'vel')
        self.update_body_scores(v_a_j = 'jerk')
    def normalize_single_pose(self,pose):
        '''divide by width,height, looks like pose is given in y,x format?'''
        ans = np.zeros(pose.shape)
        ans[:,:,0] = 3*pose[:,:,0]/self.height
        ans[:,:,1] = 3*pose[:,:,1]/self.width
        return ans
    def normalize_pose_frame(self):
        if self.isEmpty:
            self.npose = self.pose
        else:
            self.npose = [self.normalize_single_pose(i) for i in self.pose]
    def score_function(self,input,noise = 0.3):
        # make more elaborate later (hyperbolic tangent)
        return np.tanh(input)/self.fps if np.tanh(input)>noise else 0
    def score_body_part(self,body_part,how = 'accel', physics = False, noise = 0.2):
        # possible values for how : vel, accel, jerk
        # weight by score inside or outside function ...
        if self.isEmpty:
            return 0
        elif how == 'vel':
            v_vec = self.npose[-1][:,body_part,:] - self.npose[-2][:,body_part,:]
            return np.linalg.norm(v_vec) if physics else self.score_function(np.linalg.norm(v_vec),noise = noise)
        elif how == 'accel':
            v_vec = self.npose[-1][:,body_part,:] - self.npose[-2][:,body_part,:]
            a_vec = self.npose[-1][:,body_part,:] -2*self.npose[-2][:,body_part,:]+self.npose[-3][:,body_part,:]
            a_value = np.linalg.norm(a_vec)*np.mean(self.scores[-1][:,body_part]) # weight by confidence
            # return value of accel conditioned on value of vel
            return 0 if np.linalg.norm(v_vec)< 0.25 else self.score_function(np.linalg.norm(a_vec), noise = noise) # weight by confidence
        elif how == 'jerk':
            j_vec = self.npose[-1][:,body_part,:] - self.npose[-4][:,body_part,:]
            return np.linalg.norm(j_vec) if physics else self.score_function(np.linalg.norm(j_vec), noise = noise)
    def score_body(self,section = 'arms',v_a_j_option = 'accel'):
        # section can be arms/legs/torso
        dd = {'arms':[self.arms,0.2], 'legs':[self.legs,0.2], 'torso':[self.torso,0.2]}
        ans = sum([self.score_body_part(body_part,how = v_a_j_option,noise = dd[section][1]) for body_part in dd[section][0]])
        return ans
    def update_body_scores(self, v_a_j = 'accel'):
        self.arms_score[v_a_j] += self.score_body(section='arms',v_a_j_option = v_a_j)
        self.legs_score[v_a_j] += self.score_body(section='legs',v_a_j_option = v_a_j)
        self.torso_score[v_a_j] += self.score_body(section='torso',v_a_j_option = v_a_j)


class lag:
    def __init__(self, n = 1, obj = None):
        self.numItems = n
        self.items = [obj]*n
        if not obj:
            self.isEmpty = True
        else:
            self.isEmpty = False

    def push(self,new):
        if self.isEmpty == True:
            self.items = [new]*self.numItems
            self.isEmpty = False
        else:
            self.items = self.items[1:]+[new]







class lag2:
    def __init__(self,obj = None, body_part_setting = None):
        self.lst = [obj,obj,obj]
        self.confidence = [obj,obj,obj] # push the scores
        self.body_part_setting = body_part_setting #
        if not obj:
            self.isEmpty = True
        else:
            self.isEmpty = False
    def push(self,new):
        if self.isEmpty == True:
            self.lst = [new]*3
            self.isEmpty = False
        else:
            self.lst = [self.lst[1],self.lst[2],new]
    def _noisekill(self,value,noise):
        return value if value>noise else 0
    def velocity(self,noise = 30):
        if self.isEmpty==True:
            return 0
        else:
            return self._noisekill(np.linalg.norm(self.lst[2] - self.lst[1]),noise)
    def acceleration(self,noise = 30):
        if self.isEmpty == True:
            return 0
        else:
            return self._noisekill(np.linalg.norm(self.lst[2]-2*self.lst[1]+self.lst[0]),noise)

def index_of_max(lst):
    return lst.index(max(lst))
def score_and_color_from(mylag2, physics = 'accel'):
    for_total = [mylag2.arms_score[physics], mylag2.torso_score[physics], mylag2.legs_score[physics]]
    delta = [mylag2.score_body(section='arms',v_a_j_option = physics),
    mylag2.score_body(section='torso',v_a_j_option = physics),
    mylag2.score_body(section='legs',v_a_j_option = physics)
    ]
    total = sum(for_total)
    idx = index_of_max(delta)
    colors = {0:(0,255,255),1:(255,255,0),2:(0,0,255)}
    return total, colors[idx], mylag2.score_body(section='torso',v_a_j_option = physics)
def get_single_angle(v1,v2):
    '''cos(angle) = v1@v2 /|v1||v2|'''
    # normalize
    v1 = v1/np.linalg.norm(v1)
    v2 = v2/np.linalg.norm(v2)
    return np.arccos(v1@v2)*180/np.pi # units of degrees


def draw_angles(img, scores, coords, limit=3):
    '''take image and set of scores and draw it'''
    for i in range(len(scores)):
        if scores[i]==0.:
            continue
        # draw the angles
        ALT_PARTS = [posenet.PART_IDS[i] for i in ['leftShoulder','rightElbow','leftHip','rightKnee']]
        ALT_JOINTS = {k:posenet.JOINT_MAP_INDEX[k] for k in ALT_PARTS }

        for joint,ends in ALT_JOINTS.items():
            end1 = ends[0]
            end2 = ends[1]
            # vector points from joint to end
            v1 = coords[i,end1,:] - coords[i,joint,:]
            v2 = coords[i,end2,:] - coords[i,joint,:]

            angle = get_single_angle(v1,v2)
            # round based on limit
            angle = (angle//limit)*limit
            cv_position = (int(coords[i,joint,1]),int(coords[i,joint,0])) # note the opposite order
            desired_text = str(round(angle,1))

            cv2.putText(
             img, #numpy array on which text is written
             desired_text, #text
             cv_position, #position at which writing has to start
             cv2.FONT_HERSHEY_SIMPLEX, #font family
             1.5, #font size
             (0,255,255), #font color BGR
              4) #font stroke
    return img




def score_skel_and_kp(
        img, instance_scores,
        keypoint_scores,
        keypoint_coords,
        min_pose_score=0.1,
        min_part_score=0.1,
        mylag2 = lag2(), deltaLag = lag(10), draw = True):

    out_img = img


    adjacent_keypoints = []
    cv_keypoints = []
    for ii, score in enumerate(instance_scores):
        if score < min_pose_score:
            continue

        new_keypoints = get_adjacent_keypoints(
            keypoint_scores[ii, :], keypoint_coords[ii, :, :], min_part_score)
        adjacent_keypoints.extend(new_keypoints)

        # TODO
        # add condition to change color based on confidence
        for ks, kc in zip(keypoint_scores[ii, :], keypoint_coords[ii, :, :]):
            if ks < min_part_score:
                continue
            cv_keypoints.append(cv2.KeyPoint(kc[1], kc[0], 30. * ks**2)) #3rd argument represents the size, exagagerate with **2

    if draw == 1:
        out_img = cv2.drawKeypoints(
            out_img, cv_keypoints, outImage=np.array([]), color=(255, 255, 0),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        out_img = cv2.polylines(out_img, adjacent_keypoints, isClosed=False, color=(255, 255, 0), thickness = 4)


    # this is where I started to add stuff
    out_img = cv2.flip(out_img, 1) # flip so it looks more natural
    # add text to image

    for i,phy in enumerate(['accel']):
        desired_text = f"{round(score_and_color_from(mylag2, f'{phy}')[0],1)} POINTS!!!"
        cv2.putText(
         out_img, #numpy array on which text is written
         desired_text, #text
         (300, 200+100*i), #position at which writing has to start
         cv2.FONT_HERSHEY_SIMPLEX, #font family
         3, #font size
         score_and_color_from(mylag2,f'{phy}')[1], #font color
         8) #font stroke




    return out_img
