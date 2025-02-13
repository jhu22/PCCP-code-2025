import numpy as np
smi_ = np.load('uvsmi1000.npy')

import deepchem as dc
generator = dc.utils.ConformerGenerator(max_conformers=1)
coulomb_mat_eig = dc.feat.CoulombMatrixEig(max_atoms=23)

from rdkit import Chem
MolData = []
for smile in smi_:
    MolData.append(Chem.MolFromSmiles(smile))

cm_ =[]
for item in MolData:
    if item == None:
        cm_.append(np.zeros((1, 23)))
    else:
        butane_mol = generator.generate_conformers(item)
        features = coulomb_mat_eig([butane_mol])
        if features.size == 0:
            cm_.append(np.zeros((1, 23)))
        else:
            cm_.append(features)

 np.save('my_cm.npy', cm_)
