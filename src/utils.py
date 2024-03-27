

NEAR_ZERO = 1e-6

NonMetals = 'H C N O F P S Cl Se Br I'.split()
AlkaliMetals = 'Li Na K Rb Cs Fr'.split()
AlkaliEarthMetals = 'Be Mg Ca Sr Ba Ra'.split()
TransitionMetals = 'Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg'.split()
PostTransitionMetals = 'Al Ga In Sn Tl Pb Bi Po'.split()
Metalloids = 'B Si Ge As Sb Te At'.split()
Lanthanoids = 'La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu'.split()
Actinoids = 'Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr'.split()
Halogens = 'He Ne Ar Kr Xe Rn'.split()
Unknown = 'Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split()

MetalElements = AlkaliMetals + AlkaliEarthMetals + TransitionMetals + PostTransitionMetals + Metalloids + Lanthanoids + Actinoids
FunctionalGroupElements = NonMetals

