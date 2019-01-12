
data_dir = './input/'

i2label = [
    'Nucleoplasm',
    'Nuclear membrane',
    'Nucleoli',
    'Nucleoli fibrillar center',
    'Nuclear speckles',
    'Nuclear bodies',
    'Endoplasmic reticulum',
    'Golgi apparatus',
    'Peroxisomes',
    'Endosomes',
    'Lysosomes',
    'Intermediate filaments',
    'Actin filaments',
    'Focal adhesion sites',
    'Microtubules',
    'Microtubule ends',
    'Cytokinetic bridge',
    'Mitotic spindle',
    'Microtubule organizing center',
    'Centrosome',
    'Lipid droplets',
    'Plasma membrane',
    'Cell junctions',
    'Mitochondria',
    'Aggresome',
    'Cytosol',
    'Cytoplasmic bodies',
    'Rods & rings'
]

label2i = {l:i for i,l in enumerate(i2label)}