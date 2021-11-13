import pathlib
import shutil

import numpy as np

import simulation
import geometry



def copy_case(case_path) -> None :
    ''' case_template をコピーする '''
    dir_path = case_path.parent
    case_tamplate_path = dir_path.joinpath("case_template")
    remove_dir(case_path)
    shutil.copytree(case_tamplate_path, case_path)


def remove_dir(dir_path):
    ''' ディレクトリの削除 '''
    if dir_path.exists():
        shutil.rmtree(dir_path)


def save_post_data(dir_path, data_ndarray, data_info):
    ''' シミュレーション結果の保存 '''
    # ディレクトリが無ければ作成
    if not dir_path.exists():
        dir_path.mkdir()

    filename = "{}_vx={}_vy={}".format(*data_info)
    file_path = dir_path.joinpath(filename)
    np.save(file_path, data_ndarray)  # .npy  同名ファイルは上書き



if __name__ == '__main__':
    samples = 5  # Number of data generated
    
    # Pathの設定
    dir_path = pathlib.Path(__file__).parent
    case_path = dir_path.joinpath("case")
    train_data_path = dir_path.joinpath("train_data")

    # インスタンス作成
    sim = simulation.Sim(case_path)
    stl = geometry.AirfoilStl(case_path)


    for n in range(samples):
        copy_case(case_path)

        vx, vy = sim.set_U()
        sim.set_probe()

        airfoil_name = stl.create_airfoil_stl()


        if not sim.run_simulation():
            continue

        post_dat = sim.read_post_data(vx, vy)

        
        save_post_data(train_data_path, post_dat, (airfoil_name, vx, vy))



        print("{}_vx={}_vy={}".format(airfoil_name, vx, vy))

        # DEBUG
        break
