
FROM quay.io/fenicsproject/stable:2016.1.0

USER root

RUN sudo apt-get update && sudo apt-get -y install git

RUN git clone -b dolfin-adjoint-2016.1.0 https://bitbucket.org/dolfin-adjoint/dolfin-adjoint
RUN git clone -b libadjoint-2016.1.0 https://bitbucket.org/dolfin-adjoint/libadjoint

RUN cd libadjoint && mkdir build && cd build && cmake .. && sudo make install && cd ../..

RUN cd dolfin-adjoint && sudo python setup.py install && cd ..

RUN git clone https://bitbucket.org/finsberg/mesh_generation
RUN cd mesh_generation && sudo python setup.py install && cd ..

RUN git clone https://bitbucket.org/finsberg/pulse_adjoint
RUN cd pulse_adjoint && sudo python setup.py install && cd ..
