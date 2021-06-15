FROM python:3.7-slim-buster

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

#STEP 1 INSTALL THE IBM CPLEX OTIMISER

# Where to install (this is also specified in install.properties)
ARG COSDIR=/opt/CPLEX

# Default Python version is 3.7
ARG CPX_PYVERSION=3.7

# Remove stuff that is typically not needed in a container, such as IDE,
# documentation, examples. Override with --build-arg CPX_KEEP_XXX=TRUE.
ARG CPX_KEEP_IDE=FALSE
ARG CPX_KEEP_DOC=FALSE
ARG CPX_KEEP_EXAMPLES=FALSE

# Copy installer and installer arguments from local disk
COPY cplex_installer/ILOG_COS_20.10_LINUX_X86_64.bin /tmp/installer
COPY cplex_installer/install.properties /tmp/install.properties
RUN chmod u+x /tmp/installer

# Install Java runtime. This is required by the installer
RUN dpkg --configure -a
RUN apt install --fix-broken
RUN mkdir /usr/share/man/man1/
RUN apt-get update && apt-get install -y default-jre

# Install COS
RUN /tmp/installer -f /tmp/install.properties

# Remove installer, temporary files, and the JRE we installed
RUN rm -f /tmp/installer /tmp/install.properties
RUN apt-get remove -y --purge default-jre && apt-get -y --purge autoremove

RUN if [ "${CPX_KEEP_}" != TRUE ]; then rm -rf ${COSDIR}/opl/oplide; fi
RUN if [ "${CPX_KEEP_DOC}" != TRUE ]; then rm -rf ${COSDIR}/doc; fi
RUN if [ "${CPX_KEEP_EXAMPLES}" != TRUE ]; then rm -rf ${COSDIR}/*/examples; fi

# Put all the binaries (cplex/cpo interactive, oplrun) onto the path
ENV PATH ${PATH}:${COSDIR}/cplex/bin/x86-64_linux
ENV PATH ${PATH}:${COSDIR}/cpoptimizer/bin/x86-64_linux
ENV PATH ${PATH}:${COSDIR}/opl/bin/x86-64_linux

# Put the libraries onto the path
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${COSDIR}/cplex/bin/x86-64_linux
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${COSDIR}/cpoptimizer/bin/x86-64_linux
ENV LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:${COSDIR}/opl/bin/x86-64_linux

# Setup Python
ENV PYTHONPATH ${PYTHONPATH}:${COSDIR}/cplex/python/${CPX_PYVERSION}/x86-64_linux
ENV CPX_PYVERSION ${CPX_PYVERSION}


# STEP 2 INSTALL THE COTS LOCAL PYTHON PACKAGE



RUN apt-get -qq update \
    && DEBIAN_FRONTEND=noninteractive apt-get -qq install -y --no-install-recommends \
    python-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean -y \
    && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/* /tmp/* /var/tmp/*



# copy the cots package
ADD . /rac
WORKDIR /rac

# remove the cplex_installer dir which got copied with everything
RUN rm -r /rac/cplex_installer

# install the cots package
# shellcheck disable=DL3013
RUN pip3 install . \
    && apt -y purge python-dev build-essential \
    && apt -y autoremove



# Default user is cplex
RUN adduser --disabled-password --gecos "" cplex
USER cplex
WORKDIR /home/cplex

# run a shell
CMD /bin/bash


