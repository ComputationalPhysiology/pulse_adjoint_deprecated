# System imports
from distutils.core import setup
import platform
import sys
from os.path import join as pjoin

# Version number
major = 1
minor = 0


setup(name = "pulse_adjoint",
      version = "{0}.{1}".format(major, minor),
      description = """
      An adjointable cardiac mechanics solver and data assimilator.
      """,
      author = "Henrik Finsberg",
      author_email = "henriknf@simula.no",
      packages = ["pulse_adjoint",
                  "pulse_adjoint.models",
                  "pulse_adjoint.postprocess",
                  "pulse_adjoint.patient_data",
                  'pulse_adjoint.example_meshes',
                  "pulse_adjoint.unloading"],
      package_data={'pulse_adjoint.example_meshes':  ["*.h5"]},
      package_dir = {"pulse_adjoint": "pulse_adjoint"},
      )
