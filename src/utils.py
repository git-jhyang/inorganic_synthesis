

NEAR_ZERO = 1e-6

NonMetalElements = 'H C N O F P S Cl Se Br I'.split()
HalogenElements = 'He Ne Ar Kr Xe Rn Og'.split()
AlkaliElements = 'Li Na K Rb Cs Fr Be Mg Ca Sr Ba Ra'.split()
MetalElements = 'Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg'.split()
SemimetalElements = 'B Al Si Ga Ge As In Sn Sb Te Tl Pb Bi Po At'.split()
LaFamilyElements = 'La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu'.split()
AcFamilyElements = 'Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr'.split()
OtherElements = 'Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts'.split()


FunctionalGroupElements = NonMetalElements + HalogenElements
TargetGroupElements = AlkaliElements + MetalElements + SemimetalElements + LaFamilyElements + AcFamilyElements

