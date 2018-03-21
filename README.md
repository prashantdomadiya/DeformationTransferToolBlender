# DeformationTransferToolBlender

# Future improvement
It will be improvised soon for non-identical source and traget meshes (have different resolutions) .... 

# Dependency
You must install following libreries for Python 3,
1) Numpy 
2) Scipy

# Cite the Paper
Prashant Domadiya, Pratik Shah and Suman K Mitra, "Vector Graph Representation for Deformation Transfer Using Poisson Interpolation", in IEEE Winter Conference on Application of Computer Vision (WACV18), South Lake Tahoe, USA, March 2018.

# Any Query
Drop your mail at pmdomadiya@gmail.com

# Installation in Blender

1) Download the "DefromationTransferUsingVGPI.py" and save it at any directory
2) Go to File > Use Preferences > Addons > Development
3) Click on "Install from File.." button at bottom of Blender User Preferences
4) Go to the directory where you store the "DefromationTransferUsingVGPI.py" file
5) Activate this Addon by clicking right square box and close the Blender User Preferences. You can see the 3 buttons at left side in "Tools" or "Misc" in 3D view.

# How to Use?

Note: Import Mesh/Skeleton sequences in order 

1) Import Source sequence and the target poses
2) Select Source poses and click on "source seq" button
3) Select target poses and click on "target seq" button

4) If you have connection link structure of skeleten as separate file then provide path in "face path" box as ".txt" file else keep it "Default".
5) Then click on "Preprocessing" button to initialize process

6) Click on "DTPI" button to get target sequence.

# Note: Step 4 and 5 require only if you replace and edit the source and the target  
