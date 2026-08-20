import os
import SimpleITK as sitk
from tqdm import tqdm

ROOT = r"./work/temp_resampled_fixed"
OUTPUT = r"./work/temp_registered"

os.makedirs(OUTPUT, exist_ok=True)

def register_rigid(fixed, moving):

    # ---- CAST TO FLOAT ----
    fixed = sitk.Cast(fixed, sitk.sitkFloat32)
    moving = sitk.Cast(moving, sitk.sitkFloat32)

    registration = sitk.ImageRegistrationMethod()

    registration.SetMetricAsMattesMutualInformation(50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.2)

    registration.SetInterpolator(sitk.sitkLinear)

    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=200,
        gradientMagnitudeTolerance=1e-6
    )

    registration.SetOptimizerScalesFromPhysicalShift()

    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration.SetInitialTransform(initial_transform, inPlace=False)

    registration.SetShrinkFactorsPerLevel([4,2,1])
    registration.SetSmoothingSigmasPerLevel([2,1,0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration.Execute(fixed, moving)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(final_transform)

    return resampler.Execute(moving)

for patient in tqdm(os.listdir(ROOT)):

    p_path = os.path.join(ROOT, patient)
    if not os.path.isdir(p_path):
        continue

    out_path = os.path.join(OUTPUT, patient)
    os.makedirs(out_path, exist_ok=True)

    fixed = sitk.ReadImage(os.path.join(p_path,"T1.nii.gz"))
    sitk.WriteImage(fixed, os.path.join(out_path,"T1.nii.gz"))

    for mod in ["T2","FLAIR","T1CE"]:

        moving = sitk.ReadImage(os.path.join(p_path,f"{mod}.nii.gz"))
        registered = register_rigid(fixed, moving)

        sitk.WriteImage(
            registered,
            os.path.join(out_path,f"{mod}.nii.gz")
        )
