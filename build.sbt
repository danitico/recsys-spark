version := "1.0"
scalaVersion := "2.12.20"
organization := "com.recsys"
name := "recsys-spark"

unmanagedBase := baseDirectory.value / "lib"

val sparkVersion = "3.5.4"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "com.github.nscala-time" %% "nscala-time" % "2.32.0",
  "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly()
)

assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}
