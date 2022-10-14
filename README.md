# recsys-spark

A implementation in Scala of CF, Content Based, Sequential and hybrid recommender systems for Spark.

## How can you install the dependencies of this project?

### Java & Scala

We will use sdkman to install java and scala.

```bash
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"
```

Then, we will install java and scala.

```bash
sdk install scala 2.12.15
sdk install java 18.0.2-open
```

### Apache Spark and Apache Hadoop

Run the following commands:

```bash
wget https://archive.apache.org/dist/spark/spark-3.2.1/spark-3.2.1-bin-hadoop3.2.tgz
tar -xvf spark-3.2.1-bin-hadoop3.2.tgz
mv spark-3.2.1-bin-hadoop3.2 /opt
```

After that, add the following lines to .bashrc or .zshrc

```bash
export SPARK_HOME=/opt/spark-3.2.1-bin-hadoop3.2
export HADOOP_HOME=/opt/spark-3.2.1-bin-hadoop3.2
```

And apply those changes in your configuration running:

```bash
source $HOME/<your_configuration_file>
```

## How can you run this project?

First compile the project:

```bash
sbt compile
```

After that, you can run the Main.scala file with;

```bash
sbt run
```
