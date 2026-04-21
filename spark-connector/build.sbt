// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0
//
// sbt build for the FluxCompress Spark DataSource V2 connector.
//
// This project is intentionally outside the Cargo workspace so that
// `cargo build` stays fast and the Rust crates are not coupled to a
// Scala toolchain.  Users that want the connector build it separately
// with `sbt package`, then put the produced JAR plus the `flux_jni`
// cdylib on their Spark classpath / `java.library.path`.

ThisBuild / organization := "io.fluxcompress"
ThisBuild / version      := "0.1.0"
ThisBuild / scalaVersion := "2.12.18"

// Spark 3.5 is the current LTS line.  We compile against `provided`
// artefacts — Spark supplies its own copies at runtime.
val sparkVersion = "3.5.1"

// Arrow version must match what Spark 3.5 ships.
val arrowVersion = "15.0.0"

lazy val sparkConnector = (project in file("."))
  .settings(
    name := "flux-spark-connector",

    // Java 8 bytecode for the broadest Spark compatibility.
    javacOptions ++= Seq("-source", "1.8", "-target", "1.8"),
    scalacOptions ++= Seq(
      "-deprecation",
      "-feature",
      "-unchecked",
      "-encoding", "UTF-8",
      "-target:jvm-1.8",
    ),

    libraryDependencies ++= Seq(
      "org.apache.spark"  %% "spark-sql"        % sparkVersion % Provided,
      "org.apache.spark"  %% "spark-catalyst"   % sparkVersion % Provided,
      "org.apache.arrow"  %  "arrow-vector"     % arrowVersion,
      "org.apache.arrow"  %  "arrow-memory-netty" % arrowVersion,

      // Test dependencies.
      "org.scalatest"     %% "scalatest"        % "3.2.18"    % Test,
      "org.apache.spark"  %% "spark-sql"        % sparkVersion % Test,
      "org.apache.spark"  %% "spark-catalyst"   % sparkVersion % Test,
    ),

    // Make the `java/` directory in the repo root visible so the
    // generated JAR contains `FluxNative`.  Users who already have the
    // FluxNative class on their classpath can strip this out.
    Compile / unmanagedSourceDirectories += baseDirectory.value / ".." / "java",

    // The JNI cdylib lives under `target/release/` of the Cargo build.
    // Propagate it into test JVM forks so `System.loadLibrary(\"flux_jni\")`
    // works in `sbt test`.
    Test / fork               := true,
    Test / javaOptions        ++= {
      val cargoTarget = (baseDirectory.value / ".." / "target" / "release").getAbsolutePath
      Seq(s"-Djava.library.path=$cargoTarget")
    },

    // Relax Java 17+ module access so Arrow + Spark can use reflection.
    Test / javaOptions ++= Seq(
      "--add-opens=java.base/java.lang=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.invoke=ALL-UNNAMED",
      "--add-opens=java.base/java.lang.reflect=ALL-UNNAMED",
      "--add-opens=java.base/java.io=ALL-UNNAMED",
      "--add-opens=java.base/java.net=ALL-UNNAMED",
      "--add-opens=java.base/java.nio=ALL-UNNAMED",
      "--add-opens=java.base/java.util=ALL-UNNAMED",
      "--add-opens=java.base/java.util.concurrent=ALL-UNNAMED",
      "--add-opens=java.base/java.util.concurrent.atomic=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED",
      "--add-opens=java.base/sun.nio.cs=ALL-UNNAMED",
      "--add-opens=java.base/sun.security.action=ALL-UNNAMED",
      "--add-opens=java.base/sun.util.calendar=ALL-UNNAMED",
    ),
  )
