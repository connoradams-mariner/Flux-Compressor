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

// Maven groupId — based on the `datamariners.com` domain the author
// controls. Sonatype policy requires the groupId reverse the FQDN of
// a domain you own, so `com.datamariners.fluxcompress` is our
// authoritative coordinate on Maven Central.
ThisBuild / organization := "com.datamariners.fluxcompress"
ThisBuild / organizationName := "Data Mariners"
ThisBuild / organizationHomepage := Some(url("https://datamariners.com"))
ThisBuild / homepage := Some(url("https://github.com/connoradams-mariner/Flux-Compressor"))
ThisBuild / licenses := List(
  "Apache-2.0" -> url("https://www.apache.org/licenses/LICENSE-2.0")
)
ThisBuild / developers := List(
  Developer(
    id    = "connoradams-mariner",
    name  = "Connor Adams",
    email = "connor@example.com",
    url   = url("https://github.com/connoradams-mariner"),
  )
)
ThisBuild / scmInfo := Some(
  ScmInfo(
    url("https://github.com/connoradams-mariner/Flux-Compressor"),
    "scm:git:git@github.com:connoradams-mariner/Flux-Compressor.git",
  )
)

// Version is overridden by the release-maven workflow via
// `set ThisBuild / version      := "0.5.4"`; the default here only matters
// for local development.
ThisBuild / version      := "0.6.5" // x-release-please-version
ThisBuild / scalaVersion := "2.12.18"
ThisBuild / crossScalaVersions := Seq("2.12.18", "2.13.12")

// Maven Central routing — Sonatype Central Portal (central.sonatype.com).
//
// The new Portal replaced the legacy OSSRH (`s01.oss.sonatype.org`)
// for accounts created after March 2024.  `sbt-sonatype` 3.11+ speaks
// the Portal's new upload endpoint via `sonatypeCentralHost`.  The
// staging-profile-name concept still applies: it must equal the
// verified TLD root (`com.datamariners`), not the full coordinate.
ThisBuild / publishTo := sonatypePublishToBundle.value
ThisBuild / publishMavenStyle := true
ThisBuild / sonatypeCredentialHost := Sonatype.sonatypeCentralHost
ThisBuild / sonatypeProfileName := "com.datamariners"
ThisBuild / credentials += Credentials(
  "Sonatype Nexus Repository Manager",
  "s01.oss.sonatype.org",
  sys.env.getOrElse("SONATYPE_USERNAME", ""),
  sys.env.getOrElse("SONATYPE_PASSWORD", ""),
)
ThisBuild / pgpPassphrase := sys.env.get("PGP_PASSPHRASE").map(_.toCharArray)

// Spark 3.5 is the current LTS line.  We compile against `provided`
// artefacts — Spark supplies its own copies at runtime.
val sparkVersion = "3.5.1"

// Arrow version must match what Spark 3.5 ships.
val arrowVersion = "15.0.0"

lazy val sparkConnector = (project in file("."))
  .settings(
    name        := "flux-spark-connector",
    // Maven POM requires a human-readable description for any
    // artefact published to Sonatype Central Portal.
    description := "Spark DataSource V2 connector for FluxCompress — " +
      "read and write `.flux` tables from Spark 3.5 with " +
      "`df.write.format(\"flux\")`. Bundles the `flux_jni` native " +
      "library for Linux x86_64 / aarch64, macOS x64 / arm64, and " +
      "Windows x64 so no DBFS sidecar is required.",

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

    // `FluxNative.java` lives at `src/main/java/com/datamariners/fluxcompress/`
    // (the default sbt Java source path) so sbt + zinc pick it up
    // automatically. No manual `unmanagedSourceDirectories` hack
    // required.

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
