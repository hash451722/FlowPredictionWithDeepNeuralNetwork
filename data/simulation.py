import pathlib
import subprocess

import numpy as np



class Sim():
    def __init__(self, case_path, x_domain=(-1, 1), y_domain=(-1, 1), resolution=(128, 128), z_coord=0.01) -> None:
        self.case_path = case_path
        self.x_domain=x_domain
        self.y_domain=y_domain
        self.resolution = resolution

        self.x_coord = np.linspace(self.x_domain[0], self.x_domain[1], self.resolution[0])
        self.y_coord = np.linspace(self.y_domain[0], self.y_domain[1], self.resolution[1])
        self.z_coord = z_coord

        self.probe_points = None


    def set_U(self, velocity=(0.1, 10), angle=(-22.5, 22.5)):
        ''' caseファイルの設定 '''
        vx, vy = self.pick_flow_condition(velocity, angle)

        U_path = self.case_path.joinpath("0", "U")
        with open(U_path, mode="r", encoding='utf-8') as f:
            data_lines = f.read()

        data_lines = data_lines.replace("velocity_x", vx)
        data_lines = data_lines.replace("velocity_y", vy)

        with open(U_path, mode="w", encoding='utf-8') as f:
            f.write(data_lines)
        
        return vx, vy

    def pick_flow_condition(self, velocity=(0.1, 10), angle=(-22.5, 22.5)):
        ''' 
            velocity : m/s
            angle : degree
        '''
        mag = np.random.uniform(velocity[0], velocity[1]) 
        deg  = np.random.uniform(angle[0], angle[1])
        rad = np.deg2rad(deg)

        vx =  np.cos(rad) * mag
        vy = -np.sin(rad) * mag

        # 小数点以下の桁を揃える
        vx_str = str( int(vx*1000)/1000 )
        vy_str = str( int(vy*1000)/1000 )

        return vx_str, vy_str



    def set_probe(self):
        points_str = self.set_probe_points()

        probe_path = self.case_path.joinpath("system", "internalProbes")
        with open(probe_path, mode="r", encoding='utf-8') as f:
            data_lines = f.read()

        data_lines = data_lines.replace("<points>", points_str)
        data_lines = data_lines.replace("<fieldNames>", "p U")

        with open(probe_path, mode="w", encoding='utf-8') as f:
            f.write(data_lines)


    def set_probe_points(self):
        ''' データ取得座標 '''
        self.x_coord = np.linspace(self.x_domain[0], self.x_domain[1], self.resolution[0])
        self.y_coord = np.linspace(self.y_domain[0], self.y_domain[1], self.resolution[1])
        x, y = np.meshgrid(self.x_coord, self.y_coord)
        x = x.flatten()
        y = y.flatten()
        z = np.full(x.size, self.z_coord)
        self.probe_points = np.stack([x, y, z], 1)
        
        # print(self.probe_points.shape)

        s = "\n"
        for i in range(self.probe_points.shape[0]):
            s += "({} {} {})\n".format(self.probe_points[i][0], self.probe_points[i][1], self.probe_points[i][2])
        return s



    def run_simulation(self) -> bool:
        ''' [Allrun] の実行 '''
        sh_path = str( self.case_path.joinpath("Allrun.sh") )
        subprocess.run(["bash", sh_path])
        return True


    def read_post_data(self, fsx, fsy):
        ''' 出力された後処理結果の読み取り '''
        post_dir_path = self.case_path.joinpath("postProcessing", "internalProbes")

        # 最後に出力されたディレクトリを特定する
        post_dir_list = sorted(list( map( lambda p:  int(p.stem)  , list(post_dir_path.glob("*"))  ) ))
        post_dir_path = post_dir_path.joinpath(str(post_dir_list[-1]))

        post_p_path = post_dir_path.joinpath("points_p.xy")
        post_U_path = post_dir_path.joinpath("points_U.xy")

        # データの読み取り
        post_p_dat = np.loadtxt(post_p_path)  # points_p.xy : x y z p
        post_U_dat = np.loadtxt(post_U_path)  # points_U.xy : x y z vx vy vz

        post_data = np.zeros((6, self.resolution[0], self.resolution[1]))
        for dat in post_p_dat:
            idx_x = np.argmin( np.absolute(self.x_coord - dat[0]) )
            idx_y = np.argmin( np.absolute(self.y_coord - dat[1]) )
            post_data[0][idx_x][idx_y] = 1          # [0] binary mask for shape boundary
            post_data[1][idx_x][idx_y] = float(fsx) # [1] freestream field x + shape boundary
            post_data[2][idx_x][idx_y] = float(fsy) # [2] freestream field y + shape boundary
            post_data[3][idx_x][idx_y] = dat[3]     # [3] pressure

        for dat in post_U_dat:
            idx_x = np.argmin( np.absolute(self.x_coord - dat[0]) )
            idx_y = np.argmin( np.absolute(self.y_coord - dat[1]) )
            post_data[4][idx_x][idx_y] = dat[3]     # [4] velocity x
            post_data[5][idx_x][idx_y] = dat[4]     # [5] velocity y

        return post_data



if __name__ == '__main__':
    case_path = pathlib.Path(__file__).parent.joinpath("case_template")
    test = Sim(case_path)

    test.read_post_data()

    # sim.probe_points(resolution=(4,4))
    # post.set_probe(None)