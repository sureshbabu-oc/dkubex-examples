
# Build an image that can serve mlflow models.
FROM ubuntu:18.04

RUN apt-get -y update --fix-missing
RUN apt-get install -y --no-install-recommends          wget          curl          nginx          ca-certificates          bzip2          build-essential          cmake          openjdk-8-jdk          git-core          maven     && rm -rf /var/lib/apt/lists/*


# Setup miniconda
RUN curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh >> miniconda.sh
RUN bash ./miniconda.sh -b -p /miniconda && rm ./miniconda.sh
ENV PATH="/miniconda/bin:$PATH"



ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV GUNICORN_CMD_ARGS="--timeout 60 -k gevent"
# Set up the program in the image
WORKDIR /opt/mlflow

RUN pip install mlflow==1.27.0
RUN mvn --batch-mode dependency:copy -Dartifact=org.mlflow:mlflow-scoring:1.27.0:pom -DoutputDirectory=/opt/java
RUN mvn --batch-mode dependency:copy -Dartifact=org.mlflow:mlflow-scoring:1.27.0:jar -DoutputDirectory=/opt/java/jars
RUN cp /opt/java/mlflow-scoring-1.27.0.pom /opt/java/pom.xml
RUN cd /opt/java && mvn --batch-mode dependency:copy-dependencies -DoutputDirectory=/opt/java/jars


ARG MODEL_PATH
COPY $MODEL_PATH/decision-tree-classifier/ /opt/ml/model
RUN echo 'import yaml\n\
with open(r"/opt/ml/model/conda.yaml") as file:\n\
    f = yaml.load(file, Loader=yaml.FullLoader)\n\
for index, item in enumerate(f["dependencies"]):\n\
    if type(item) is dict and item.get("pip") != None:\n\
        f["dependencies"][index]["pip"].append("protobuf==3.19.4")\n\
        with open(r"/opt/ml/model/conda.yaml", "w") as file:\n\
            yaml.dump(f, file)' >> /tmp/update_conda_yaml.py
RUN python /tmp/update_conda_yaml.py

RUN python -c \
        'from mlflow.models.container import _install_pyfunc_deps;\
        _install_pyfunc_deps(\
        "/opt/ml/model", \
        install_mlflow=False, \
        enable_mlserver=False, \
        env_manager="conda")'

ENV MLFLOW_DISABLE_ENV_CREATION="true"
ENV ENABLE_MLSERVER=False
                

# granting read/write access and conditional execution authority to all child directories 
# and files to allow for deployment to AWS Sagemaker Serverless Endpoints 
# (see https://docs.aws.amazon.com/sagemaker/latest/dg/serverless-endpoints.html)
RUN chmod o+rwX /opt/mlflow/

ENTRYPOINT ["python", "-c", "from mlflow.models import container as C;C._serve('conda')"]
