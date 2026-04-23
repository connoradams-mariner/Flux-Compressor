// Plugins for the Spark connector's release pipeline.
//
//   * `sbt-pgp`       — signs the published artefact with our PGP key
//                       (Maven Central rejects unsigned uploads).
//   * `sbt-sonatype`  — drives the OSSRH staging / release dance so
//                       the artefact surfaces at
//                       `com.datamariners.fluxcompress:flux-spark-connector_2.12`.
//
// Both plugins are invoked by `.github/workflows/release-maven.yml`;
// local development doesn't need them but keeping them in the plugin
// tree means the release job doesn't pull surprise versions.

addSbtPlugin("com.github.sbt"      % "sbt-pgp"      % "2.2.1")
// 3.11+ adds Sonatype Central Portal support (`sonatypeCentralHost`,
// `sonatypeBundleRelease` routing through the Portal API).
addSbtPlugin("org.xerial.sbt"      % "sbt-sonatype" % "3.11.3")
