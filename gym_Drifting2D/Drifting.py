import gym
from gym.envs.classic_control import rendering
import numpy as np
import math
import shapely
from shapely.geometry import LineString, Point
from gym.envs.registration import register

walls = []

walls.append([240, 809, 200, 583])
walls.append([200, 583, 218, 395])
walls.append([218, 395, 303, 255])
walls.append([303, 255, 548, 173])
walls.append([548, 173, 764, 179])
walls.append([764, 179, 1058, 198])
walls.append([1055, 199, 1180, 215])
walls.append([1177, 215, 1220, 272])
walls.append([1222, 273, 1218, 367])
walls.append([1218, 367, 1150, 437])
walls.append([1150, 437, 1044, 460])
walls.append([1044, 460, 757, 600])
walls.append([757, 600, 1099, 570])
walls.append([1100, 570, 1187, 508])
walls.append([1187, 507, 1288, 443])
walls.append([1288, 443, 1463, 415])
walls.append([1463, 415, 1615, 478])
walls.append([1617, 479, 1727, 679])
walls.append([1727, 679, 1697, 874])
walls.append([1694, 872, 1520, 964])
walls.append([1520, 964, 1100, 970])
walls.append([1105, 970, 335, 960])
walls.append([339, 960, 264, 899])
walls.append([263, 897, 238, 803])
walls.append([317, 782, 274, 570])
walls.append([275, 569, 284, 407])
walls.append([284, 407, 363, 317])
walls.append([363, 317, 562, 240])
walls.append([562, 240, 1114, 284])
walls.append([1114, 284, 1120, 323])
walls.append([1120, 323, 1045, 377])
walls.append([1045, 378, 682, 548])
walls.append([682, 548, 604, 610])
walls.append([604, 612, 603, 695])
walls.append([605, 695, 702, 713])
walls.append([703, 712, 1128, 642])
walls.append([1129, 642, 1320, 512])
walls.append([1323, 512, 1464, 497])
walls.append([1464, 497, 1579, 535])
walls.append([1579, 535, 1660, 701])
walls.append([1660, 697, 1634, 818])
walls.append([1634, 818, 1499, 889])
walls.append([1499, 889, 395, 883])
walls.append([395, 883, 330, 838])
walls.append([330, 838, 315, 782])
walls.append([319, 798, 306, 725])
walls.append([276, 580, 277, 543])
walls.append([603, 639, 622, 590])
walls.append([599, 655, 621, 704])
walls.append([1074, 571, 1115, 558])
walls.append([1314, 516, 1333, 511])
walls.append([1692, 875, 1706, 830])
walls.append([277, 912, 255, 872])
walls.append([1214, 262, 1225, 288])
walls.append([1601, 470, 1625, 490])
walls.append([1119, 644, 1139, 634])
walls.append([687, 710, 719, 710])
walls.append([1721, 664, 1727, 696])
walls.append([1015, 392, 1065, 362])
walls.append([1091, 572, 1104, 568])
walls.append([1157, 528, 1233, 478])

map = walls
reward_gates = [[613, 268, 613, 156], [546, 272, 465, 168], [483, 298, 368, 179], [411, 316, 301, 248], [363, 342, 231, 306], [324, 393, 189, 381], [299, 447, 189, 473], [291, 517, 187, 568], [305, 585, 213, 647], [213, 710, 325, 708], [222, 816, 352, 772], [260, 927, 359, 840], [361, 971, 416, 858], [475, 979, 490, 852], [578, 980, 578, 880], [643, 979, 646, 869], [718, 984, 713, 870], [778, 979, 787, 887], [852, 978, 876, 877], [958, 983, 972, 867], [1040, 976, 1051, 883], [1095, 977, 1126, 860], [1159, 983, 1191, 871], [1222, 980, 1240, 877], [1284, 973, 1297, 877], [1367, 980, 1374, 884], [1452, 975, 1445, 883], [1540, 967, 1507, 873], [1626, 929, 1577, 822], [1716, 835, 1630, 771], [1733, 736, 1646, 703], [1618, 667, 1716, 602], [1598, 611, 1681, 526], [1547, 554, 1597, 441], [1467, 528, 1495, 423], [1392, 529, 1370, 422], [1323, 541, 1256, 450], [1261, 575, 1175, 493], [1155, 642, 1087, 525], [1025, 678, 1026, 557], [923, 699, 930, 569], [807, 707, 841, 600], [701, 711, 746, 627], [611, 657, 720, 591], [719, 509, 809, 571], [862, 542, 813, 480], [932, 521, 919, 445], [1030, 473, 966, 378], [1113, 454, 1065, 364], [1215, 386, 1102, 330], [1099, 298, 1225, 260], [1047, 287, 1087, 191], [949, 288, 958, 187], [856, 284, 854, 179], [761, 275, 759, 167]]

downScaleFactor = 1.1

for i in range(len(map)):
    for j in range(len(map[i])):
        map[i][j] = map[i][j]/downScaleFactor
        map[i][0] -= 20
        map[i][2] -= 20

for i in range(len(reward_gates)):
    for j in range(len(reward_gates[i])):
        reward_gates[i][j] = reward_gates[i][j]/downScaleFactor
        reward_gates[i][0] -= 20
        reward_gates[i][2] -= 20


class Drifting(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(self):
        self.viewer = None
        self.pos = [650, 200]
        self.velX = 0
        self.velY = 0
        self.drag = 0.9
        self.angularVel = 0.0
        self.angularDrag = 0.6
        self.power = 0.7
        self.turnSpeed = 0.04
        self.angle = math.radians(-90)
        self.w = 10
        self.h = 20
        self.on = 0

        self.states = 12
        self.actions = 9

    def step(self, action):

        if (action == 0):
            self.acc()
        if (action == 1):
            self.decc()
        if (action == 2):
            self.left()
        if (action == 3):
            self.right()

        if (action==4):
            self.acc()
            self.left()
        if (action==5):
            self.acc()
            self.right()

        if (action==6):
            self.decc()
            self.left()
        if (action==7):
            self.decc()
            self.right()


        self.pos[0] += self.velX
        self.pos[1] += self.velY

        self.velX *= self.drag
        self.velY *= self.drag
        self.angle += self.angularVel
        self.angularVel *= self.angularDrag
        reward = -0.01

        ded = False

        for i in map:
            if self.checkCol(i):
                ded = True
                break

        if (self.checkCol(reward_gates[self.on])):
            self.on += 1
            reward = 1

        if (self.on > len(reward_gates) - 1):
            self.on = 0

        if (ded):
            reward = -1
        state = self.getState()

        return state, reward, ded, None

    def render(self):
        if self.viewer is None:
            self.viewer = rendering.Viewer(1500, 1000)
            LIST = [[5, 10, 5, -10], [-5, 10, -5, -10], [5, 10, -5, 10], [5, -10, -5, -10]]
            verts = []
            for i in LIST:
                LINE = self.rotatePos(i[0], i[1], i[2], i[3], self.angle)

                self.viewer.draw_line([LINE[0], LINE[1]], [LINE[2], LINE[3]], color=(255, 0, 0))
                verts.append((LINE[0], LINE[1]))
                verts.append((LINE[2], LINE[3]))

            CAR = rendering.make_polygon(verts)
            self.viewer.add_geom(CAR)

            for LINE in map:
                Line = rendering.make_polyline([(LINE[0], LINE[1]), (LINE[2], LINE[3])])
                Line.set_linewidth(5)
                self.viewer.add_geom(Line)



        LIST = [[5, 10, 5, -10], [-5, 10, -5, -10], [5, 10, -5, 10], [5, -10, -5, -10]]
        verts = []
        for i in LIST:
            LINE = self.rotatePos(i[0], i[1], i[2], i[3], self.angle)
            verts.append((LINE[0], LINE[1]))
            verts.append((LINE[2], LINE[3]))



        CAR = rendering.make_polygon(verts)
        self.viewer.geoms[0] = CAR




        self.viewer.render()

    def reset(self):
        self.pos = [650, 200]
        self.velX = 0
        self.velY = 0
        self.drag = 0.9
        self.angularVel = 0.0
        self.angularDrag = 0.6
        self.power = 0.7
        self.turnSpeed = 0.04
        self.angle = math.radians(-90)
        self.w = 10
        self.h = 20
        self.on = 0
        return self.getState()

    def acc(self):
        self.velX += math.sin(self.angle) * self.power
        self.velY += math.cos(self.angle) * self.power

        if (self.velX > 10):
            self.velX = 10

        if (self.velY > 10):
            self.velY = 10

    def decc(self):
        self.velX -= math.sin(self.angle) * self.power
        self.velY -= math.cos(self.angle) * self.power

        if (self.velX < -10):
            self.velX = -10

        if (self.velY < -10):
            self.velY = -10

    def right(self):
        self.angularVel += self.turnSpeed

    def left(self):
        self.angularVel -= self.turnSpeed

    def LineInter(self, L1, L2):
        A = (L1[0], L1[1])
        B = (L1[2], L1[3])

        # line 2
        C = (L2[0], L2[1])
        D = (L2[2], L2[3])

        line1 = LineString([A, B])
        line2 = LineString([C, D])

        int_pt = line1.intersection(line2)

        if not int_pt.is_empty:
            point_of_intersection = int_pt.x, int_pt.y
            return point_of_intersection

        return False

    def checkCol(self, line_):
        LIST = [[5, 10, 5, -10], [-5, 10, -5, -10], [5, 10, -5, 10], [5, -10, -5, -10]]
        l = []
        coll = False
        for i in LIST:
            LINE = self.rotatePos(i[0], i[1], i[2], i[3], self.angle)
            inter = self.LineInter(LINE, line_)

            # pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
            #                      ("v2f", (LINE[0], LINE[1], LINE[2], LINE[3]))
            #                      , ('c3B', [50, 255, 30] * 2))
            if not inter == False:
                coll = True

        return coll

    def rotatePos(self, offW, offH, offW2, offH2, angle):
        x = self.pos[0]
        y = self.pos[1]

        X = x + offW
        Y = y + offH

        New_X = x + (X - x) * math.cos(-angle) - (Y - y) * math.sin(-angle)

        New_Y = y + (X - x) * math.sin(-angle) + (Y - y) * math.cos(-angle)

        X = x + offW2
        Y = y + offH2

        New_X2 = x + (X - x) * math.cos(-angle) - (Y - y) * math.sin(-angle)
        New_Y2 = y + (X - x) * math.sin(-angle) + (Y - y) * math.cos(-angle)

        return [New_X, New_Y, New_X2, New_Y2]

    def getState(self):

        LIST = [[0, 0, 1000, 0], [0, 0, -1000, 0], [0, 0, -1000, 1000], [0, 0, 0, 1000], [0, 0, 1000, 1000]]
        DS = []
        bongs = []
        for i in LIST:
            closest = 10000000
            closesIN = []
            LINE = self.rotatePos(i[0], i[1], i[2], i[3], self.angle)
            for LI in map:

                inter = self.LineInter(LINE, LI)

                if not inter == False:
                    distS = math.dist([LINE[0], LINE[1]], [inter[0], inter[1]])
                    if (distS < closest):
                        closest = distS
                        closesIN = inter

            if (len(closesIN) > 0):
                # pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
                #                      ("v2f", (LINE[0], LINE[1], closesIN[0], closesIN[1]))
                #                      , ('c3B', [255, 255, 255] * 2))
                if (not self.viewer is None):
                    self.viewer.draw_line([LINE[0], LINE[1]], [closesIN[0], closesIN[1]])
                bongs.append([LINE[0], LINE[1], closesIN[0], closesIN[1]])
                DS.append(closest)
            else:
                # pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
                #                      ("v2f", (LINE[0], LINE[1], LINE[2], LINE[3]))
                #                      , ('c3B', [255, 255, 0] * 2))
                bongs.append([LINE[0], LINE[1], LINE[2], LINE[3]])
                DS.append(-1)

        # ---------------
        e = [reward_gates[self.on]]
        for i in bongs:
            closest = 10000000
            closesIN = []
            LINE = i
            for LI in e:

                inter = self.LineInter(LINE, LI)

                if not inter == False:
                    distS = math.dist([LINE[0], LINE[1]], [inter[0], inter[1]])
                    if (distS < closest):
                        closest = distS
                        closesIN = inter

            if (len(closesIN) > 0):
                # pyglet.graphics.draw(2, pyglet.gl.GL_LINES,
                #                      ("v2f", (LINE[0], LINE[1], closesIN[0], closesIN[1]))
                #                      , ('c3B', [255, 255, 0] * 2))
                DS.append(closest)
            else:

                DS.append(-1)

        DS.append(self.angularVel)
        vector = np.array([self.velX, self.velY])

        magnitude = np.linalg.norm(vector)
        DS.append(magnitude)
        DS = np.array(DS)
        norm = np.linalg.norm(DS)
        normal_array = DS / norm

        return normal_array


register(
    id='CarDrifting2D-v0',
    entry_point='gym_Drifting2D.Drifting:Drifting'
)