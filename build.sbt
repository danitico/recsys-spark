scalaVersion := "2.12.15"

name := "recommendations"
version := "1.0"

unmanagedBase := baseDirectory.value / "lib"

libraryDependencies ++= Seq(
  "org.scala-lang" % "scala-library" % scalaVersion.value,
  "org.apache.spark" %% "spark-sql" % "3.2.1",
  "org.apache.spark" %% "spark-mllib" % "3.2.1",
  "org.apache.spark" %% "spark-core" % "3.2.1"
)
