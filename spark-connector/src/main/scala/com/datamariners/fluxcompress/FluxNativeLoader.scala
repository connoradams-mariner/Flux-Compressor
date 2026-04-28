// Copyright 2024 FluxCompress Contributors
// SPDX-License-Identifier: Apache-2.0

package com.datamariners.fluxcompress

import java.io.{File, InputStream}
import java.nio.file.{Files, Path, StandardCopyOption}

/**
 * Boot-strap loader that extracts the bundled `flux_jni` cdylib from
 * the connector JAR and registers it with the JVM via
 * [[System.load]].
 *
 * This object is invoked from a static initialiser in `FluxNative` so
 * that the first JNI call in a user's Spark job transparently wires
 * up the native library — no `java.library.path` configuration,
 * no DBFS sidecar uploads, no driver/executor classpath gymnastics.
 *
 * The layout inside the JAR mirrors the Maven-release workflow's
 * output:
 *
 * {{{
 * /native/
 *   linux-x86_64/libflux_jni.so
 *   linux-aarch64/libflux_jni.so
 *   darwin-x86_64/libflux_jni.dylib
 *   darwin-aarch64/libflux_jni.dylib
 *   windows-x86_64/flux_jni.dll
 * }}}
 *
 * A user who wants to override the bundled library (to run a locally
 * patched build, for example) can either:
 *   1. Set `-Dflux.native.path=/abs/path/to/libflux_jni.so` on their
 *      Spark driver/executor command line, or
 *   2. Put a matching cdylib on `java.library.path` and set
 *      `-Dflux.native.loadlib=true` to use `System.loadLibrary` instead.
 */
private[fluxcompress] object FluxNativeLoader {

  /** Named path to a user-supplied cdylib (overrides bundled resources). */
  private val PropPath    = "flux.native.path"
  /** When true, fall back to `System.loadLibrary("flux_jni")`. */
  private val PropLoadLib = "flux.native.loadlib"

  @volatile private var loaded: Boolean = false

  /** Idempotently load the native library. Throws on failure. */
  def ensureLoaded(): Unit = {
    if (loaded) return
    this.synchronized {
      if (loaded) return
      doLoad()
      loaded = true
    }
  }

  private def doLoad(): Unit = {
    // 1. Explicit path wins.
    Option(System.getProperty(PropPath)) match {
      case Some(p) if p.nonEmpty =>
        System.load(new File(p).getAbsolutePath)
        return
      case _ =>
    }

    // 2. `loadLibrary` opt-in: resolves via `java.library.path`.
    if ("true".equalsIgnoreCase(System.getProperty(PropLoadLib, "false"))) {
      System.loadLibrary("flux_jni")
      return
    }

    // 3. Default: extract the bundled resource for this OS + arch.
    val resource = bundledResourcePath()
    val stream   = Option(getClass.getResourceAsStream(resource))
      .getOrElse(
        throw new UnsatisfiedLinkError(
          s"FluxCompress does not ship a native library for ${osArchDescription()}. " +
          s"Set -D$PropPath=/abs/path/to/flux_jni to point at a local build, " +
          s"or run `cargo build --release -p flux-jni-bridge` and set " +
          s"-D$PropLoadLib=true with -Djava.library.path=$$CARGO_TARGET/release."
        )
      )
    val tempPath = extractToTemp(stream, resource)
    System.load(tempPath.toAbsolutePath.toString)
  }

  /**
   * Return the resource path inside the JAR that holds the cdylib for
   * the current OS + architecture.
   */
  private def bundledResourcePath(): String = {
    val osName = System.getProperty("os.name", "").toLowerCase
    val osArch = System.getProperty("os.arch", "").toLowerCase

    val osDir = if (osName.contains("linux"))      "linux"
                else if (osName.contains("mac"))   "darwin"
                else if (osName.contains("windows")) "windows"
                else throw new UnsatisfiedLinkError(s"unsupported OS: $osName")

    val archDir = osArch match {
      case "amd64" | "x86_64"              => "x86_64"
      case "aarch64" | "arm64"             => "aarch64"
      case other                           =>
        throw new UnsatisfiedLinkError(s"unsupported arch: $other")
    }

    val filename = osDir match {
      case "linux"   => "libflux_jni.so"
      case "darwin"  => "libflux_jni.dylib"
      case "windows" => "flux_jni.dll"
    }

    s"/native/$osDir-$archDir/$filename"
  }

  private def osArchDescription(): String =
    s"os=${System.getProperty("os.name")} arch=${System.getProperty("os.arch")}"

  /** Copy a resource stream to a temp file with the right extension. */
  private def extractToTemp(in: InputStream, resource: String): Path = {
    val dot = resource.lastIndexOf('.')
    val suffix = if (dot >= 0) resource.substring(dot) else ""
    // Use `flux-native-` + hash so parallel JVMs don't collide.
    val tmp = Files.createTempFile("flux-native-", suffix)
    tmp.toFile.deleteOnExit()
    try {
      Files.copy(in, tmp, StandardCopyOption.REPLACE_EXISTING)
    } finally {
      in.close()
    }
    tmp
  }
}
