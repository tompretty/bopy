from setuptools import Distribution, setup

Distribution().fetch_build_eggs(["numpy"])

setup(
    name="bopy",
    author="Tom Pretty",
    author_email="tpretty@robots.ox.ac.uk",
    license="MIT",
    packages=["bopy"],
    setup_requires=["numpy"],
    install_requires=[
        "numpy",
        "sobol-seq",
        "pydoe",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "gpy",
        "scipydirect",
    ],
    zip_safe=False,
)
