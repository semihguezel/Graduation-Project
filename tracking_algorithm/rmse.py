import numpy as np

def rmse(rot_coord, tip, cm_dg, coords_webot, coords_calculated):

    MSE = np.subtract(coords_webot,coords_calculated)
    MSE_X = np.subtract(coords_webot[:,0],coords_calculated[:,0])
    MSE_Y = np.subtract(coords_webot[:,1],coords_calculated[:,1])
    MSE_Z = np.subtract(coords_webot[:,2],coords_calculated[:,2])

    print(f"-------------------------------------------------------------------------- {tip} METHOD ({rot_coord})")
    if rot_coord == "coordinates":
        x_or_roll = "X-Y-Z"
        MSE = 100 * MSE
        MSE_X = 100 * MSE_X
        MSE_Y = 100 * MSE_Y
        MSE_Z = 100 * MSE_Z
    
    else:
        x_or_roll = "ROLL-PITCH-YAW"

    print(f"Max errors for {x_or_roll}: [{max(abs(MSE_X)):,.2f} {max(abs(MSE_Y)):,.2f} {max(abs(MSE_Z)):,.2f}] {cm_dg}")

    MSE = np.square(MSE).mean() 
    MSE_X = np.square(MSE_X).mean() 
    MSE_Y = np.square(MSE_Y).mean() 
    MSE_Z = np.square(MSE_Z).mean() 

    RMSE_X = np.sqrt(MSE_X)
    RMSE_Y = np.sqrt(MSE_Y)
    RMSE_Z = np.sqrt(MSE_Z)

    RMSE = np.sqrt(MSE)

    print(f"Total Root Mean Square Error: {RMSE:,.2f} {cm_dg} \nfor X: {RMSE_X:,.2f} {cm_dg} \nfor Y: {RMSE_Y:,.2f} {cm_dg} \nfor Z: {RMSE_Z:,.2f} {cm_dg}")

def run_rmse():
    coords_webot = np.loadtxt("txt/coords_360.txt")
    length = int(len(coords_webot) / 3)
    coords_webot = coords_webot.reshape(length, 3)
    rot_webot = np.loadtxt("txt/rot_360.txt").reshape(length, 3)
    rot_webot = (-rot_webot * 180) /np.pi


    for i in range(len(coords_webot)):
        coords_webot[i][0] = coords_webot[i][0] + 5
        coords_webot[i][1] = coords_webot[i][1] + 5
        coords_webot[i][2] = coords_webot[i][2] + 4

    coords_calculated_L_BFGS_B = np.loadtxt("txt/coords_calculated_L-BFGS-B_coord.txt").reshape(length,3)
    coords_calculated_BFGS = np.loadtxt("txt/coords_calculated_BFGS_coord.txt").reshape(length,3)

    rot_calculated_L_BFGS_B = np.loadtxt("txt/coords_calculated_L-BFGS-B_rot.txt").reshape(length,3)
    rot_calculated_BFGS = np.loadtxt("txt/coords_calculated_BFGS_rot.txt").reshape(length,3)

    rmse_dict = {1:{"rot_coord":"coordinates", "tip":"L_BFGS_B", "cm_dg":"cm", "coords_webot":coords_webot ,"coords_calculated":coords_calculated_L_BFGS_B},
                2:{"rot_coord":"rotation", "tip":"L_BFGS_B", "cm_dg":"deg", "coords_webot":rot_webot, "coords_calculated":rot_calculated_L_BFGS_B},
                3:{"rot_coord":"coordinates", "tip":"BFGS", "cm_dg":"cm", "coords_webot":coords_webot, "coords_calculated":coords_calculated_BFGS},
                4:{"rot_coord":"rotation", "tip":"BFGS", "cm_dg":"deg", "coords_webot":rot_webot,"coords_calculated":rot_calculated_BFGS}}
    for i in rmse_dict:
        rmse(rmse_dict[i]["rot_coord"],rmse_dict[i]["tip"],rmse_dict[i]["cm_dg"],rmse_dict[i]["coords_webot"],rmse_dict[i]["coords_calculated"])

#run_rmse()