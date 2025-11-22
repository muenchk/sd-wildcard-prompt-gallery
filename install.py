import launch

if not launch.is_installed("dynamicprompts"):
    launch.run_pip("install dynamicprompts", "requirement for Wildcard Gallery")

if not launch.is_installed("numpy"):
    launch.run_pip("install numpy", "requirement for Wildcard Gallery")