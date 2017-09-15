import os
import traceback

import paddle.v2 as paddle
from flask import Flask, jsonify, request
from flask_cors import CORS
from Queue import Queue
import threading

tarfn = os.getenv('PARAMETER_TAR_PATH', None)

if tarfn is None:
    raise ValueError(
        "please specify parameter tar file path with environment variable PARAMETER_TAR_PATH"
    )

topology_filepath = os.getenv('TOPOLOGY_FILE_PATH', None)

if topology_filepath is None:
    raise ValueError(
        "please specify topology file path with environment variable TOPOLOGY_FILE_PATH"
    )

with_gpu = os.getenv('WITH_GPU', '0') != '0'

port = int(os.getenv('PORT', '80'))

app = Flask(__name__)
CORS(app)


def errorResp(msg):
    return jsonify(code=-1, message=msg)


def successResp(data):
    return jsonify(code=0, message="success", data=data)


sendQ = Queue()
recvQ = Queue()


@app.route('/', methods=['POST'])
def infer():
    sendQ.put(request.json)
    success, resp = recvQ.get()
    if success:
        return successResp(resp)
    else:
        return errorResp(resp)


# PaddlePaddle v0.10.0 does not support inference from different
# threads, so we create a single worker thread.
def worker():
    paddle.init(use_gpu=with_gpu)
    with open(tarfn) as param_f, open(topology_filepath) as topo_f:
        params = paddle.parameters.Parameters.from_tar(param_f)
        inferer = paddle.inference.Inference(parameters=params, fileobj=topo_f)

    while True:
        j = sendQ.get()
        try:
            feeding = {}
            d = []
            for i, key in enumerate(j):
                d.append(j[key])
                feeding[key] = i
                r = inferer.infer([d], feeding=feeding)
        except:
            trace = traceback.format_exc()
            recvQ.put((False, trace))
            continue
        recvQ.put((True, r.tolist()))


if __name__ == '__main__':
    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()
    print 'serving on port', port
    app.run(host='0.0.0.0', port=port, threaded=True)
