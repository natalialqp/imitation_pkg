import numpy as np

class HT(object):
    '''
    This class is in charge of calculating the homogeneous transformation
    '''
    @staticmethod
    def Rx(theta):
        '''
        This function performs the rotation about x
        Input:
        theta -> angle to be rotated

        Output:
        R -> resultant rotation
        '''
        R = np.array([[1., 0., 0.],
                      [0., np.cos(theta), -np.sin(theta)],
                      [0., np.sin(theta), np.cos(theta)]])
        return R

    @staticmethod
    def Ry(theta):
        '''
        This function performs the rotation about y
        Input:
        theta -> angle to be rotated

        Output:
        R -> resultant rotation
        '''
        R = np.array([[np.cos(theta),  0., np.sin(theta)],
                      [0., 1., 0.],
                      [-np.sin(theta), 0., np.cos(theta)]])
        return R

    @staticmethod
    def Rz(theta):
        '''
        This function performs the rotation about z
        Input:
        theta -> angle to be rotated

        Output:
        R -> resultant rotation
        '''
        R = np.array([[np.cos(theta), -np.sin(theta), 0.],
                     [np.sin(theta), np.cos(theta),  0.],
                     [0., 0., 1.]])
        return R

    @staticmethod
    def get_homogeneous_transform(euler_rotation, translation_vector):
        '''
        This function performs the get_homogeneous transformation
        Input:
        euler_rotation -> angles to be rotated according to x, y and z axis
        translation_vector -> distances to be translated in x, y and z

        Output:
        T -> resultantant transformation
        '''
        gamma = euler_rotation[0]
        beta = euler_rotation[1]
        alpha = euler_rotation[2]
        rotation_matrix = HT.Rz(alpha).dot(HT.Ry(beta)).dot(HT.Rx(gamma))

        t = np.array(translation_vector)[np.newaxis].T
        T = np.hstack((rotation_matrix, t))
        T = np.vstack((T, np.array([0., 0., 0., 1.])))
        return T

    @staticmethod
    def cam_frame_to_base_frame(element, is_vector):
        '''
        This function performs the get_homogeneous transformation for an element from camera frame to base frame
        Input:
        element -> vector or point
        is_vector -> bool

        Output:
        position_base_link -> resultantant transformation
        '''
        if is_vector:
            element.append(1)
        else:
            if element.shape == (3, 3):
                t = np.array([0., 0., 0.,])[np.newaxis].T
                element = np.hstack((element, t))
                element = np.vstack((element, np.array([0., 0., 0., 1.])))
        cam_base_link_translation = [0.094, 0., 0.5962]
        cam_base_link_rot = [np.pi/2, 0., np.pi/2]
        cam_base_link_tf = HT.get_homogeneous_transform(cam_base_link_rot, cam_base_link_translation)
        position_base_link = cam_base_link_tf.dot(element)
        if is_vector:
            position_base_link = position_base_link.flatten()[0:3]
        return position_base_link

    @staticmethod
    def get_translation(matrix):
        '''
        This function returns the translation given in a matrix
        Input:
        matrix -> HT

        Output:
        translation vector
        '''
        return matrix[:,3][:3]
