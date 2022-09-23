from setuptools import setup, find_packages
setup(
    name="melkor_parser",
    version="1.0",
    author="Yiqi Sun (Zongjing Li)",
    author_email="ysun697@gatech.edu",
    description="Melkor Program Parser.",

    # project main page
    url="http://jiayuanm.com/", 

    # the package that are prerequisites
    packages=find_packages(),
    package_data={
        '':['melkor_parser'],
        'bandwidth_reporter':['melkor_parser']
               },
)