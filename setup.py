from setuptools import setup, find_namespace_packages

setup(
    name="diffusion_deep_dream_research",
    version="0.1.0",
    packages=find_namespace_packages(include=["diffusion_deep_dream_research*", "submodules*"]),
)