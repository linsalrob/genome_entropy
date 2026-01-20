"""Basic tests to verify package installation and structure."""

import genome_entropy


def test_version():
    """Test that version is defined."""
    assert hasattr(genome_entropy, "__version__")
    assert isinstance(genome_entropy.__version__, str)


def test_package_import():
    """Test that the package can be imported."""
    assert genome_entropy is not None
