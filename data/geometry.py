import pathlib

import numpy as np
import matplotlib.pyplot as plt


class AirfoilStl():
    def __init__(self, case_path) -> None:
        self.case_path = case_path
        self.database_path = None
        self.airfoil_list = None
        self.init_path_list()

    def init_path_list(self) -> None:
        dir_path = pathlib.Path(__file__).parent
        self.database_path = dir_path.joinpath("airfoil_database")
        self.airfoil_list = list(self.database_path.glob("*.dat"))


    def create_airfoil_stl(self, airfoil_name="random"):
        p = self.select_airfoil(airfoil_name)
        dat = self.format_dat(p)
        s = create_stl_solid(dat, "airfoil")

        stl_file_path = self.case_path.joinpath("constant", "geometry", "airfoil.stl")
        stl_file_path.write_text(s)

        return p.stem


    def select_airfoil(self, airfoil_name="random"):
        ''' ファイルの絶対パスを返す '''
        if airfoil_name == "random":
            i = np.random.randint(0, len(self.airfoil_list))
            return self.airfoil_list[i]
        else:
            al = list( map(lambda x: x.stem, self.airfoil_list) )
            if airfoil_name in al:
                k = al.index(airfoil_name)
                return self.airfoil_list[k]
            else:
                raise RuntimeError('[{}] does not exist.'.format(airfoil_name))
                

        
    def format_dat(self, airfoil_path):
        airfoil_dat = np.loadtxt(airfoil_path, skiprows=1)  # <class 'numpy.ndarray'>

        # 閉じた形状にする
        clearance = np.sqrt( np.sum( np.square(airfoil_dat[0]-airfoil_dat[-1]) ) )
        if clearance > 1e-6:
            airfoil_dat = np.vstack([ airfoil_dat, airfoil_dat[0]] )

        return airfoil_dat
    

    def draw_airfoil(self, airfoil_name="random"):
        path = self.select_airfoil(airfoil_name)
        dat = self.format_dat(path)

        fig, ax = plt.subplots()
        ax.plot(dat[:, 0], dat[:,1])
        ax.grid()
        ax.axis('equal')
        ax.set_title(path.stem, fontsize=10)
        plt.show()




def create_stl_solid(dat, solid_name="walls") -> str:
    ''' 2次元座標からSTLの文字列を作成 '''
    if dat.shape[1]  != 2:
        return False

    z = 1
    s = "solid {}\n".format(solid_name)
    for i in range(dat.shape[0]-1):
        v0 = np.hstack( (dat[i],   0) )
        v1 = np.hstack( (dat[i+1], 0) )
        v2 = np.hstack( (dat[i],   z) )
        s += facet( v0, v1, v2)

        v0 = np.hstack( (dat[i],   z) )
        v1 = np.hstack( (dat[i+1], z) )
        v2 = np.hstack( (dat[i+1], 0) )
        s += facet( v0, v1, v2)

    s += "endsolid {}".format(solid_name)
    return s


def facet(v0, v1, v2) -> str:

    nx, ny, nz = normal_vector(v0, v1, v2)
    s0 = "facet normal {} {} {}\n".format(nx, ny, nz)    
    s1 = "  outer loop\n"
    s2 = "    vertex {} {} {}\n".format(v0[0], v0[1], v0[2])
    s3 = "    vertex {} {} {}\n".format(v1[0], v1[1], v1[2])
    s4 = "    vertex {} {} {}\n".format(v2[0], v2[1], v2[2])
    s5 = "  endloop\n"
    s6 = "endfacet\n"
    return s0 + s1 +s2 + s3 +s4 + s5 + s6


def normal_vector(v0, v1, v2):
    ''' 法線ベクトル '''
    vec01 = v1 - v0
    vec02 = v2 - v0
    x = vec01[1]*vec02[2] - vec01[2]*vec02[1]
    y = vec01[2]*vec02[0] - vec01[0]*vec02[2]
    z = vec01[0]*vec02[1] - vec01[1]*vec02[0]
    return x, y, z



if __name__ == '__main__':
    stl = AirfoilStl(None)

    print( len(stl.airfoil_list) )

    fig, ax = plt.subplots()
    for airfoil_path in stl.airfoil_list:

        dat = stl.format_dat(airfoil_path)

        ax.plot(dat[:, 0], dat[:,1])
        ax.grid()
        ax.axis('equal')
        # ax.set_title(path.stem, fontsize=10)
    plt.show()

    # stl.draw_airfoil()
