FROM paddlepaddle/paddle

ENV PARAMETER_TAR_PATH=/data/param.tar \
    TOPOLOGY_FILE_PATH=/data/inference_topology.pkl
ADD requirements.txt /root
ADD main.py /root
RUN pip install -r /root/requirements.txt
CMD ["python", "/root/main.py"]
