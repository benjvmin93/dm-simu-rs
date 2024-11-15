from ubuntu:22.04

ENV DOCKER_OPTS="--dns 8.8.8.8 --dns 8.8.4.4"

RUN apt-get update && \
	apt-get upgrade -y && \
	apt-get install -y bash wget gpg curl wget python3 python3-dev python3-venv build-essential vim

SHELL ["/bin/bash", "-c"]

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH=$PATH:/.cargo/bin

ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu

ENV WRKDIR=/app

RUN . $HOME/.cargo/env

ADD . $WRKDIR

EXPOSE 80

RUN python3 -m venv /env && \
	/env/bin/pip install --upgrade pip && \
	/env/bin/pip install -r $WRKDIR/python/requirements.txt

WORKDIR $WRKDIR

CMD ["/bin/bash", "-c", "source /env/bin/activate && exec /bin/bash"]
