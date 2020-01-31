"""
CDSW
====

Utilities for Python on Cloudera Data Science Workbench.
"""
import os
import shutil
import traceback
import warnings
import time
import re
import urllib
import base64
import uuid

import requests
import simplejson

try:
    # python 2
    import futures
except ImportError:
    # python 3
    import concurrent.futures as futures

# Make isinstance(my_string, basestring) work in both Python 2 and 3
try:
    basestring
except NameError:
    basestring = str

modelAuthenticationKeyPath = "/etc/model-serving-environment/authentication-key"


__all__ = ["get_auth", "launch_workers", "list_workers", "stop_workers", "await_workers",
           "track_file", "track_metric", "call_model", "read_metrics", "track_delayed_metric", 
           "track_aggregate_metric"]

THREAD_POOL_SIZE = 10

# Plan. For now, you have to:
# - parse the model deployment id out of env vars like DS_RUNTIME_FRAUD_DETECTION_1_10_SERVICE_HOST. The deployment id is the second integer.
# - Use the new monitoring service route to get the Crn
# - use the model authentication key for auth, in base64 format. The monitoring service will accept it. That's all you have.

# So, the monitoring service should take model deplomyent id or Crn, and should
# accept engine api key.

# Also, given that you won't be able to bake modifications to the model runtime in,
# require the user to call the model_metrics decorator.

def _get_model_deployment_id_in_model_replica():
    # Look for env vars like DS_RUNTIME_FRAUD_DETECTION_1_10_SERVICE_HOST.
    # The deployment id is the second integer
    for (k, v) in os.environ.items():
        m = re.search(r'\w+_\d+_(\d+)_SERVICE_HOST', k)
        if m is not None:
            deployment_id_str = m.group(1)
            return int(deployment_id_str)
    raise RuntimeError("Could not detect model deployment id")

def _is_experiment():
    """
    Return True if running in the context of a CDSW experiment.
    """
    engine_type = os.environ.get("CDSW_ENGINE_TYPE", None)
    return True if engine_type == "experiment" else False

def _is_model_replica():
    """
    Return True if running in the context of a CDSW model replica.
    """
    return os.path.islink(modelAuthenticationKeyPath)

def _get_monitoring_service_endpoint():
    """
    Detects the URL of the monitoring service.
    """
    apiUrl = os.environ.get("CDSW_API_URL")
    if apiUrl is None:
        raise RuntimeError("Environmental variable CDSW_API_URL must be set.")
    parsed = urllib.parse.urlparse(apiUrl)
    return parsed.scheme + "://monitoring." + parsed.netloc

def _get_monitoring_service_auth():
    userAPIKey = os.environ.get("CDSW_USER_API_KEY", None)
    if userAPIKey is not None:
        return {"Authorization": userAPIKey}
    if _is_model_replica():
        authKey = open(modelAuthenticationKeyPath, "rb").read()
        return {"Authorization": base64.b64encode(authKey)}
    raise RuntimeError("Please set the CDSW_USER_API_KEY environmental variable.")

def track_file(src):
    """
    Description
    -----------

    Saves the file `src` for access after a CDSW Experiment.
    """
    if not _is_experiment():
        return
    cdsw_output_dir = os.environ.get("CDSW_OUTPUT_DIR", None)
    if cdsw_output_dir is None:
        raise RuntimeError(
            "Environment variable CDSW_OUTPUT_DIR is not set. "
            "File not tracked."
        )
    if not os.path.isdir(cdsw_output_dir):
        raise OSError(
            "{} is not a directory. File not tracked".format(cdsw_output_dir)
        )
    shutil.copy(src, os.environ["CDSW_OUTPUT_DIR"])

def call_endpoint(url, data):
    """call_endpoint

    Description
    -----------

    Calls a CDSW REST API endpoint running in a different engine.

    Parameters
    ----------
    url: string
        The URL of the endpoint to call.
    data: object
        The closest Python equivalent to a JSON data structure: a dict or
        list containing strings, numbers, booleans, None or other dicts or
        lists.

    Returns
    -------
    object
        A data structure of the same form as the 'data' argument.
    """
    auth = get_auth()
    headers = {
        'Content-type': 'application/json',
        'Accept': 'application/json'
    }
    r = requests.post(
        url,
        data=simplejson.dumps(data),
        auth=(auth["user"], auth["password"]),
        headers=headers
    )
    result = r.json()
    if r.status_code == requests.codes.ok:
        return result
    else:
        warnings.warn(result["message"])

def call_model(api_key, ipt):
    """
    TODO: docstring
    """
    url = os.environ["CDSW_API_URL"].replace("/v1", "/altus-ds-1/models/call-model")
    
    if not isinstance(ipt, dict):
      raise RuntimeError("Input must be a dict")
      
    body = {"accessKey": api_key, "request": ipt}
    
    headers = {
        'Content-type': 'application/json',
        'Accept': 'application/json'
    }
    
    r = requests.post(url, data=simplejson.dumps(body), headers=headers)
    
    result = r.json()
    if r.status_code == requests.codes.ok:
      return result['response']
    else:
      print(result)
      raise RuntimeError(result["error"]["message"])

_current_metrics = {
    "startTimestampMs": 0,
    "endTimestampMs": 0,
    "userMetrics": {},
    "predictionUuid": "",
    "schemaVersion": 2
}

_model_deployment_crn = {"crn": None}

def _get_model_deployment_crn():
    if _model_deployment_crn["crn"] is not None:
        print(_model_deployment_crn["crn"])
        return _model_deployment_crn["crn"]
    if _is_model_replica():
        headers = {
            'Content-type': 'application/json',
            'Accept': 'application/json'
        }

        model_deployment_id = _get_model_deployment_id_in_model_replica()
        url = _get_monitoring_service_endpoint() + "/get-model-deployment-crn"
        request_body = {
            "modelDeploymentID": model_deployment_id
        }
        auth_hdrs = _get_monitoring_service_auth()

        headers.update(auth_hdrs)

        resp = requests.post(
            url, data=simplejson.dumps(request_body),
            headers=headers
        )
        _model_deployment_crn["crn"] = resp.json()["modelDeploymentCrn"]
        print(_model_deployment_crn["crn"])
        return _model_deployment_crn["crn"]
    else:
        raise RuntimeError("Please set environmental variable MODEL_DEPLOYMENT_Crn")

def _emit_metrics(metrics):

    headers = {
        'Content-type': 'application/json',
        'Accept': 'application/json'
    }

    endpoint = _get_monitoring_service_endpoint()
    model_deployment_crn = _get_model_deployment_crn()

    url = _get_monitoring_service_endpoint() + "/track-metrics"

    auth_hdrs = _get_monitoring_service_auth()

    headers.update(auth_hdrs)

    request_body = {
        "modelDeploymentCrn": model_deployment_crn,
        "startTimestampMs": _current_metrics["startTimestampMs"],
        "endTimestampMs": _current_metrics["endTimestampMs"],
        "predictionUuid": _current_metrics["predictionUuid"],
        "metrics": _current_metrics["userMetrics"]
    }

    resp = requests.post(url, data=simplejson.dumps(request_body), headers=headers)

    return resp.json()

def _init_metrics_for_prediction(model_crn, prediction_uuid):
    _current_metrics["userMetrics"].clear()
    _current_metrics["startTimestampMs"] = int(round(time.time() * 1000))
    _current_metrics["predictionUuid"] = prediction_uuid
    _current_metrics["modelDeploymentCrn"] = model_crn

def _finalize_metrics_for_prediction():
    _current_metrics["endTimestampMs"] = int(round(time.time() * 1000))
    _emit_metrics(_current_metrics)  

# This is a decorator, it shims missing functionality in 
# the base cdsw.py. Actually we might want to use it as
# it works for external serving layers.
def model_metrics(predict, model_crn=None, user_api_key=None, monitoring_endpoint=None):
    """
    FIXME: docstring
    """
    if model_crn is None:
        if _is_model_replica():
            model_crn = _get_model_deployment_crn()
        else:
            raise RuntimeError("Model Crn must be provided outside of a model replica.")
    if user_api_key is None:
        if os.environ.get("CDSW_USER_API_KEY") is None:
            # Don't consume the headers, just test that it works
            _get_monitoring_service_auth()
     
    def predict_wrapped(ipt):
        prediction_uuid = uuid.uuid4().__str__()
        _init_metrics_for_prediction(model_crn, prediction_uuid)
        
        out = predict(ipt)
        
        _finalize_metrics_for_prediction()
        
        return {'prediction': out, 'uuid': prediction_uuid}
    return predict_wrapped

def read_metrics(model_deployment_crn, start_timestamp_ms, end_timestamp_ms):
    headers = {
        'Content-type': 'application/json',
        'Accept': 'application/json'
    }

    endpoint = _get_monitoring_service_endpoint()

    url = _get_monitoring_service_endpoint() + "/read-metrics"

    auth_hdrs = _get_monitoring_service_auth()

    headers.update(auth_hdrs)

    request_body = {
        "modelDeploymentCrn": model_deployment_crn,
        "startTimestampMs": start_timestamp_ms,
        "endTimestampMs": end_timestamp_ms
    }

    resp = requests.post(url, data=simplejson.dumps(request_body), headers=headers)

    return resp.json()

def track_metric(key, value):
    """
    Description
    -----------

    Tracks a metric for an experiment

    Parameters
    ----------
    key: string
        The metric key to track.
    value: string, boolean, numeric
        The metric value to track.
    """
    if _is_experiment():
        if "CDSW_EXPERIMENT_ID" not in os.environ:
            raise RuntimeError(
                "Environment variable CDSW_EXPERIMENT_ID not set. "
                "Metric not tracked."
            )

        invalid_types = {dict, tuple, list}
        if type(value) in invalid_types:
            raise ValueError(
                "Metric value must not be of type {}".format(type(value).__name__)
            )
        if not isinstance(key, basestring):
            raise ValueError(
                "Metric key must be a string. "
                "{} is type {}.".format(key, type(key).__name__)
            )

        baseURL = os.environ["CDSW_DS_API_URL"] + "/trackRunMetrics"
        experimentId = os.environ["CDSW_EXPERIMENT_ID"]
        data = {"id": experimentId, "metrics": {key: value}}
        call_endpoint(baseURL, data)
    elif _is_model_replica():
        _current_metrics["userMetrics"][key] = value

def track_delayed_metrics(metrics, model_deployment_crn, prediction_uuid):
    headers = {
        'Content-type': 'application/json',
        'Accept': 'application/json'
    }

    endpoint = _get_monitoring_service_endpoint()

    url = _get_monitoring_service_endpoint() + "/track-delayed-metrics"

    auth_hdrs = _get_monitoring_service_auth()

    headers.update(auth_hdrs)

    request_body = {
        "modelDeploymentCrn": model_deployment_crn,
        "predictionUuid": prediction_uuid,
        "metrics": metrics
    }

    resp = requests.post(url, data=simplejson.dumps(request_body), headers=headers)

    return resp.json()

#delayed_metrics = {'ground_truth': decimal.Decimal('0')}
#prediction_uuid = '0da0af80-10ab-4d9d-abb6-06e24d6b1100'
#track_delayed_metrics(delayed_metrics, model_crn, prediction_uuid)
  
def track_aggregate_metrics(metrics, model_deployment_crn, start_timestamp_ms, end_timestamp_ms):
    headers = {
        'Content-type': 'application/json',
        'Accept': 'application/json'
    }

    endpoint = _get_monitoring_service_endpoint()

    url = _get_monitoring_service_endpoint() + "/track-aggregate-metrics"

    auth_hdrs = _get_monitoring_service_auth()

    headers.update(auth_hdrs)

    request_body = {
        "modelDeploymentCrn": model_deployment_crn,
        "startTimestampMs": start_timestamp_ms,
        "endTimestampMs": end_timestamp_ms,
        "metrics": metrics
    }

    resp = requests.post(url, data=simplejson.dumps(request_body), headers=headers)

    return resp.json()

def get_auth():
    """
    Description
    -----------

    Returns the username and password to use with the CDSW REST API.

    Returns
    -------
    dict
        A dict with keys 'user' and 'password'.
    """
    # FIXME: If model replica, get the authentication_key in b64 format.
    # If CDSW_USER_API_KEY is set, use that instead.
    # Use it as the Authorization header.
    if "CDSW_API_KEY" in os.environ:
        return {"user": os.environ["CDSW_API_KEY"], "password": ""}
    else:
        raise RuntimeError("Environment variable CDSW_API_KEY is not set.")


def get_master_id():
    if os.environ["CDSW_MASTER_ID"] == "":
        master_id = os.environ["CDSW_ENGINE_ID"]
    else:
        master_id = os.environ["CDSW_MASTER_ID"]
    return master_id


def get_base_url():
    return os.environ["CDSW_PROJECT_URL"] + "/dashboards"


def launch_workers(n, cpu, memory, nvidia_gpu=0, kernel="python3", script="",
                   code="", env={}):
    """
    Description
    -----------

    Launches worker engines into the cluster.

    Parameters
    ----------
    n: int
        The number of engines to launch.
    cpu: float
        The number of CPU cores to allocate to the engine.
    memory: float
        The number of gigabytes of memory to allocate to the engine.
    nvidia_gpu: int, optional
        The number of GPU's to allocate to the engine.
    kernel: str, optional
        The kernel. Can be "r", "python2", "python3" or "scala".
    script: str, optional
        The name of a Python source file the engine should execute as soon as
        it starts up.
    code: str, optional
        Python code the engine should execute as soon as it starts up. If
        script is specified, code will be ignored.
    env: dict, optional
        Environment variables to set in the engine.

    Returns
    -------
    list
        A list of dicts describing the engines.
    """

    request_body = {
        "kernel": kernel,
        "cpu": cpu,
        "memory": memory,
        "nvidia_gpu": nvidia_gpu,
        "script": script,
        "code": code,
        "env": env,
        "master_id": get_master_id()
    }
    url = get_base_url()
    auth = get_auth()

    headers = {
        'Content-type': 'application/json',
        'Accept': 'application/json'
    }

    # The n launch requests are done concurrently in a thread pool for lower
    # latency.
    def launch_worker(i):
        r = requests.post(
            url, data=simplejson.dumps(request_body),
            auth=(auth["user"], auth["password"]), headers=headers
        )
        return r.json()
    pool = futures.ThreadPoolExecutor(THREAD_POOL_SIZE)
    responses = [pool.submit(launch_worker, i) for i in range(n)]
    return [x.result() for x in futures.wait(responses)[0]]


def get_id(worker_or_id):
    try:
        basestring_ = basestring
    except NameError:
        basestring_ = str
    if isinstance(worker_or_id, basestring_):
        return worker_or_id
    else:
        return worker_or_id["id"]


def stop_workers(*ids):
    """stop_workers

    Description
    -----------

    Stops worker engines.

    Parameters
    ----------
    ids: int or list of worker descriptions, optional
        The id's of the worker engines to stop. If not provided, all
        worker engines in the cluster will be stopped.

    Returns
    -------
    list
        A list of dicts describing the engines.
    """
    if len(ids) == 0:
        ids = [worker["id"] for worker in list_workers()]
        if len(ids) == 0:
            return
        return stop_workers(*ids)
    else:
        ids_to_stop = [get_id(worker_or_id) for worker_or_id in ids]
        base_url = get_base_url()
        auth = get_auth()

        # The stop requests are done concurrently in a thread pool for
        # lower latency.
        def stop_worker(id):
            return requests.put(
                base_url + "/" + str(id) + "/stop",
                auth=(auth["user"], auth["password"])
            )

        pool = futures.ThreadPoolExecutor(THREAD_POOL_SIZE)

        responses = [pool.submit(stop_worker, id) for id in ids]
        return [x.result() for x in futures.wait(responses)[0]]


def list_workers():
    """
    Description
    -----------

    Returns all information on all the workers in the cluster.

    Returns
    -------
    list
        A list of dicts describing the engines.
    """
    auth = get_auth()
    url = os.environ["CDSW_ENGINE_URL"] + "/workers"
    return requests.get(url, auth=(auth["user"], auth["password"])).json()


def await_workers(ids, wait_for_completion=True, timeout_seconds=60):
    """await_workers

    Description
    -----------

    Waits for workers to either reach the 'running' status, or to
    complete and exit.

    Parameters
    ----------
    ids: int or list of worker descriptions, optional
        The id's of the worker engines to stop or the worker's 
        description dicts as returned by launch_workers or 
        list_workers. If not provided, all workers in the cluster 
        will be stopped.
    wait_for_completion: boolean, optional
        If True, will wait for all workers to exit successfully.
        If False, will wait for all workers to reach the 'running'
        status.
        Defaults to True.
    timeout_seconds: int, optional
        Maximum number of seconds to wait for workers to reach the
        desired status. Defaults to 60. If equal to 0, there is no
        timeout. Workers that have not reached the desired status
        by the timeout will be returned in the 'failures' key. See
        the return value documentation.

    Returns
    -------
    dict
        A dict with keys 'workers' and 'failures'. The 'workers'
        key contains a list of dicts describing the workers that
        reached the desired status. The 'failures' key contains a
        list of descriptions of the workers that did not.

        Note: If wait_for_completion is False, the workers in the
        'workers' key will contain a key called 'ip_address'
        which contains each worker's external IP address. This can be
        useful for running distributed frameworks on workers.
    """
    t = 0
    poll_interval = 5
    out = {"workers": [], "failures": []}
    running = set()
    ids_to_await = set([get_id(worker_or_id) for worker_or_id in ids])
    while timeout_seconds != 0 and t <= timeout_seconds:
        workers_now = list_workers()
        status_dict = dict([(worker['id'], worker['status']) for worker in workers_now])
        done = True
        for worker in workers_now:
            id_ = worker["id"]
            if id_ not in ids_to_await:
                continue
            if status_dict[id_] == 'failed':
                out['failures'].append(worker)
                ids_to_await.remove(id_)
            elif status_dict[id_] == 'timedout':
                out['failures'].append(worker)
                ids_to_await.remove(id_)
            elif status_dict[id_] == 'stopped':
                out['failures'].append(worker)
                ids_to_await.remove(id_)
            else:
                if wait_for_completion:
                    if status_dict[id_] == 'succeeded':
                        # If the worker has reached the 'succeeded' status,
                        # and we are waiting for completion, it is a success.
                        out['workers'].append(worker)
                        ids_to_await.remove(id_)
                    if status_dict[id_] != 'succeeded':
                        # If the worker is in a non-terminal state, and we
                        # are waiting for completion, exit the loop iteration
                        # and wait.
                        done = False
                        break
                else:
                    if status_dict[id_] == 'succeeded':
                        # We want the workers to reach 'running', so this is a
                        # failure.
                        out['failures'].append(worker)
                        ids_to_await.remove(id_)
                    elif status_dict[id_] == 'running' and worker.get("ip_address") != "unknown":
                        # If the worker has reached the 'running' status, and we
                        # are not waiting for completion, it is a success.
                        out['workers'].append(worker)
                        ids_to_await.remove(id_)
                    else:
                        # If the worker is in a non-terminal state but is not
                        # running, we need to exit the loop iteration and wait.
                        done = False
                        break
        if done:
            return out
        else:
            time.sleep(poll_interval)
            t = t + poll_interval

    # Here we have timed out. All workers that are not successes
    # are considered failures.
    if len(ids_to_await) > 0:
      workers_now = list_workers()
      for worker in workers_now:
          if worker["id"] in ids_to_await:
              out['failures'].append(worker)

    return out
