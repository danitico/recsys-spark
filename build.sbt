version := "1.0"
scalaVersion := "2.12.15"
organization := "com.recsys"
name := "recsys-spark"

unmanagedBase := baseDirectory.value / "lib"

val sparkVersion = "3.2.1"

libraryDependencies ++= Seq(
  "org.scala-lang" % "scala-library" % scalaVersion.value,
  "org.apache.spark" %% "spark-sql" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-mllib" % sparkVersion % "provided",
  "org.apache.spark" %% "spark-core" % sparkVersion % "provided",
  "com.github.nscala-time" %% "nscala-time" % "2.32.0"
)
