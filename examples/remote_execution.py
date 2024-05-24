from simple_kfp_task import Task

def run():
    from test import test
    import os 

    os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "true"

    test()


if __name__ == "__main__":
    task = Task.init(
        func=run,
        packages=[],
        namespace="workshop",
        remote_url="https://github.com/sebh96/kfp-task.git",
        container_image="tensorflow/tensorflow:2.10.1-gpu",
        memory_limit="4Gi",
        gpu_limit=1
    )

    task.run()