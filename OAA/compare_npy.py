import numpy as np

seam_file = '/esat/izar/r0833114/SEAM/results_voc/results_cam/2007_000032.npy'
oaa_file = '/esat/izar/r0833114/OAA/results_voc/exp2/results_cam/2007_000032.npy'

seam = np.load(seam_file, allow_pickle=True)
oaa = np.load(oaa_file, allow_pickle=True)

seam_dict = seam.item()
oaa_dict = oaa.item()

print('seam dict',seam_dict)
print('oaa dict', oaa_dict)
print('seam item: ',seam_dict[0].shape)
print('oaa item: ', oaa_dict[0].shape)
