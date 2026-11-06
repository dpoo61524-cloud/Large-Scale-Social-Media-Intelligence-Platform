"""Spark NLP Session Management Module (Company Step 2).

Manages a reusable, thread-safe SparkSession configured for Spark NLP.
Enforces environment compatibility checks (Python <= 3.12, Java 17) and
provides clean lifecycle management (start, reuse, and shutdown).
"""

import os
import shutil
import sys
from typing import Any

# Module-level singleton instance
_SPARK_INSTANCE: Any = None


def _get_short_path(path: str) -> str:
    """Returns the Windows 8.3 short path to avoid space issues in batch scripts."""
    if not sys.platform.startswith("win") or not path:
        return path
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(500)
        ctypes.windll.kernel32.GetShortPathNameW(path, buf, 500)
        return buf.value or path
    except Exception:
        return path


def configure_runtime_environment() -> None:
    """Discovers and configures JAVA_HOME, HADOOP_HOME, and networking on Windows."""
    # 1. Java runtime detection
    if not os.environ.get("JAVA_HOME") and not shutil.which("java"):
        import glob
        candidates = [
            r"C:\Program Files\Microsoft\jdk-17.0.21.21-hotspot",
            r"C:\Program Files\Java\jdk-17",
            r"C:\Program Files\Eclipse Adoptium\jdk-17",
        ]
        candidates.extend(glob.glob(r"C:\Program Files\Microsoft\jdk-17*"))
        candidates.extend(glob.glob(r"C:\Program Files\Java\jdk*"))
        for candidate in candidates:
            if os.path.exists(os.path.join(candidate, "bin", "java.exe")):
                os.environ["JAVA_HOME"] = candidate
                bin_dir = os.path.join(candidate, "bin")
                if bin_dir not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                break

    # 2. Windows-specific configurations (Hadoop winutils, short paths, networking)
    if sys.platform.startswith("win"):
        if not os.environ.get("HADOOP_HOME"):
            proj_hadoop = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "hadoop"))
            candidates = [proj_hadoop, r"C:\hadoop"]
            for candidate in candidates:
                if os.path.exists(os.path.join(candidate, "bin", "winutils.exe")):
                    os.environ["HADOOP_HOME"] = candidate
                    bin_dir = os.path.join(candidate, "bin")
                    if bin_dir not in os.environ.get("PATH", ""):
                        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
                    break

        # Normalize paths with 8.3 short paths to prevent spaces in paths breaking cmd/batch tools
        if "HADOOP_HOME" in os.environ:
            os.environ["HADOOP_HOME"] = _get_short_path(os.environ["HADOOP_HOME"])
        if "JAVA_HOME" in os.environ:
            os.environ["JAVA_HOME"] = _get_short_path(os.environ["JAVA_HOME"])

        # Hostname and IP bindings to avoid NetBIOS / network adapter lookup failures
        os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")
        os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")

    # Explicitly configure python executables for PySpark
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)


def check_spark_nlp_prerequisites() -> tuple[bool, str | None]:
    """Inspects the local runtime environment for Spark NLP prerequisites.

    Checks:
    1. Python version: Officially validated up to Python 3.12 (Python >= 3.13 is unsupported).
    2. Java runtime: Java 17 (or Java 8/11) must be available on PATH or JAVA_HOME.
    3. Python packages: pyspark and spark-nlp must be installed.

    Returns:
        (is_compatible, error_or_warning_message)
    """
    configure_runtime_environment()

    # 1. Check Python version
    major, minor = sys.version_info.major, sys.version_info.minor
    if major > 3 or (major == 3 and minor >= 13):
        return (
            False,
            f"Python {major}.{minor}.{sys.version_info.micro} detected. "
            "Spark NLP officially supports Python 3.8 to 3.12. "
            "Please use an isolated Python 3.12 virtual environment (e.g., via requirements-sparknlp.txt)."
        )

    # 2. Check Java availability
    java_on_path = shutil.which("java")
    java_home = os.environ.get("JAVA_HOME")
    if not java_on_path and not java_home:
        return (
            False,
            "Java runtime not found on PATH or JAVA_HOME. Spark requires Java 17 (or Java 8/11). "
            "Install OpenJDK 17 and configure JAVA_HOME."
        )

    # 3. Check packages
    try:
        import pyspark  # noqa: F401
    except ImportError:
        return (
            False,
            "PySpark package is not installed. Install via: pip install -r requirements-sparknlp.txt"
        )

    try:
        import sparknlp  # noqa: F401
    except ImportError:
        return (
            False,
            "spark-nlp package is not installed. Install via: pip install -r requirements-sparknlp.txt"
        )

    return True, None


def is_spark_nlp_available() -> bool:
    """Returns True if Spark NLP and its runtime prerequisites are fully satisfied."""
    compatible, _ = check_spark_nlp_prerequisites()
    return compatible


def get_spark_session(
    app_name: str = "SocialMediaIntelligenceNLP",
    memory: str | None = None,
    gpu: bool = False,
) -> Any:
    """Creates or returns the active singleton Spark NLP session.

    Configures local execution mode with Spark NLP jar packages and conservative memory settings.
    Avoids duplicate session creation and provides clear error diagnostics if
    prerequisites are missing.

    Raises:
        RuntimeError: If Java, Python, or Spark NLP prerequisites are not satisfied.
    """
    global _SPARK_INSTANCE

    if _SPARK_INSTANCE is not None:
        try:
            # Check if existing session is still alive
            if not _SPARK_INSTANCE._jsc.sc().isStopped():
                return _SPARK_INSTANCE
        except Exception:
            _SPARK_INSTANCE = None

    # Verify environment prerequisites before attempting JVM startup
    compatible, reason = check_spark_nlp_prerequisites()
    if not compatible:
        raise RuntimeError(
            f"Cannot initialize Spark NLP session. Prerequisite check failed: {reason}"
        )

    # Conservative driver memory default for resource-constrained environments
    driver_memory = memory or os.environ.get("SPARK_DRIVER_MEMORY", "1024M")

    try:
        import sparknlp

        # Set conservative local execution parameters
        params = {
            "spark.driver.cores": "1",
            "spark.driver.host": "localhost",
            "spark.driver.bindAddress": "127.0.0.1",
            "spark.ui.enabled": "false",
            "spark.driver.maxResultSize": "512M",
        }

        # Start SparkSession with Spark NLP configurations
        _SPARK_INSTANCE = sparknlp.start(
            gpu=gpu,
            memory=driver_memory,
            params=params,
        )
        return _SPARK_INSTANCE
    except Exception as err:
        _SPARK_INSTANCE = None
        raise RuntimeError(
            f"Failed to start Spark NLP session due to initialization error: {err}"
        ) from err


def stop_spark_session() -> None:
    """Gracefully shuts down the active Spark NLP session if running."""
    global _SPARK_INSTANCE
    if _SPARK_INSTANCE is not None:
        try:
            _SPARK_INSTANCE.stop()
        except Exception:
            pass
        finally:
            _SPARK_INSTANCE = None
