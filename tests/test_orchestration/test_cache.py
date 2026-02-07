"""
Tests for TaskCache: fingerprint computation and persistent caching.
"""
import pytest
import tempfile
from pathlib import Path
from core.orchestration.cache import (
    compute_fingerprint,
    compute_config_fingerprint,
    TaskCache,
    _normalize_payload,
)


class TestFingerprintComputation:
    """Tests for fingerprint computation functions."""
    
    def test_compute_fingerprint_basic(self):
        """Test basic fingerprint computation."""
        fp = compute_fingerprint(
            "data",
            {"instruments": ["AAPL"], "start_date": "2020-01-01"},
            []
        )
        
        assert isinstance(fp, str)
        assert len(fp) == 16  # First 16 chars of SHA256
    
    def test_compute_fingerprint_stability(self):
        """Test that same inputs produce same fingerprint."""
        payload = {"instruments": ["AAPL", "MSFT"], "start_date": "2020-01-01"}
        
        fp1 = compute_fingerprint("data", payload, [])
        fp2 = compute_fingerprint("data", payload, [])
        
        assert fp1 == fp2
    
    def test_compute_fingerprint_different_payloads(self):
        """Test that different payloads produce different fingerprints."""
        fp1 = compute_fingerprint("data", {"key": "value1"}, [])
        fp2 = compute_fingerprint("data", {"key": "value2"}, [])
        
        assert fp1 != fp2
    
    def test_compute_fingerprint_with_dependencies(self):
        """Test fingerprint with dependency fingerprints."""
        fp1 = compute_fingerprint("signals", {"config_id": "test"}, ["dep1", "dep2"])
        fp2 = compute_fingerprint("signals", {"config_id": "test"}, ["dep1", "dep3"])
        
        assert fp1 != fp2
    
    def test_normalize_payload_sorts_instruments(self):
        """Test that instrument lists are sorted for stability."""
        normalized1 = _normalize_payload({"instruments": ["AAPL", "MSFT", "GOOG"]})
        normalized2 = _normalize_payload({"instruments": ["GOOG", "AAPL", "MSFT"]})
        
        assert normalized1["instruments"] == normalized2["instruments"]
        assert normalized1["instruments"] == ["AAPL", "GOOG", "MSFT"]
    
    def test_normalize_payload_paths(self):
        """Test that Path objects are converted to strings."""
        normalized = _normalize_payload({"path": Path("/tmp/test")})
        
        assert isinstance(normalized["path"], str)
        assert normalized["path"] == "/tmp/test"
    
    def test_compute_config_fingerprint(self, tmp_path):
        """Test computing fingerprint from config file contents."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("key: value\n")
        
        fp = compute_config_fingerprint(config_file)
        
        assert isinstance(fp, str)
        assert len(fp) == 16
    
    def test_compute_config_fingerprint_stability(self, tmp_path):
        """Test that same config contents produce same fingerprint."""
        config_file = tmp_path / "test.yaml"
        config_file.write_text("key: value\n")
        
        fp1 = compute_config_fingerprint(config_file)
        fp2 = compute_config_fingerprint(config_file)
        
        assert fp1 == fp2
    
    def test_compute_config_fingerprint_missing_file(self, tmp_path):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_config_fingerprint(tmp_path / "nonexistent.yaml")
    
    def test_compute_config_fingerprint_pickle_file(self, tmp_path):
        """Test that pickle files can be fingerprinted (binary content)."""
        import pickle
        
        config_file = tmp_path / "test.pkl"
        test_data = {"key": "value", "nested": {"a": 1}}
        
        with open(config_file, "wb") as f:
            pickle.dump(test_data, f)
        
        # Should not raise UnicodeDecodeError
        fp = compute_config_fingerprint(config_file)
        
        assert isinstance(fp, str)
        assert len(fp) == 16
        
        # Same pickle content should produce same fingerprint
        fp2 = compute_config_fingerprint(config_file)
        assert fp == fp2


class TestTaskCache:
    """Tests for TaskCache persistent caching."""
    
    def test_create_cache(self, tmp_path):
        """Test creating a task cache."""
        cache = TaskCache(cache_dir=tmp_path / "cache")
        
        assert cache.cache_dir.exists()
        assert cache.hits == 0
        assert cache.misses == 0
    
    def test_put_and_get(self, tmp_path):
        """Test storing and retrieving cached results."""
        cache = TaskCache(cache_dir=tmp_path / "cache")
        
        fingerprint = "abc123def456"
        result = {"output": "path/to/result"}
        
        cache.put(fingerprint, result)
        retrieved = cache.get(fingerprint)
        
        assert retrieved == result
        assert cache.hits == 1
        assert cache.misses == 0
    
    def test_get_miss(self, tmp_path):
        """Test cache miss for non-existent fingerprint."""
        cache = TaskCache(cache_dir=tmp_path / "cache")
        
        retrieved = cache.get("nonexistent")
        
        assert retrieved is None
        assert cache.hits == 0
        assert cache.misses == 1
    
    def test_cache_persistence(self, tmp_path):
        """Test that cache persists across instances."""
        cache_dir = tmp_path / "cache"
        
        # First instance: put
        cache1 = TaskCache(cache_dir=cache_dir)
        cache1.put("fingerprint1", {"result": "data1"})
        
        # Second instance: get (should find cached value)
        cache2 = TaskCache(cache_dir=cache_dir)
        retrieved = cache2.get("fingerprint1")
        
        assert retrieved == {"result": "data1"}
    
    def test_clear_cache(self, tmp_path):
        """Test clearing all cached results."""
        cache = TaskCache(cache_dir=tmp_path / "cache")
        
        cache.put("fp1", {"data": "1"})
        cache.put("fp2", {"data": "2"})
        
        cache.clear()
        
        assert cache.get("fp1") is None
        assert cache.get("fp2") is None
        assert cache.hits == 0
        assert cache.misses == 2
    
    def test_get_stats(self, tmp_path):
        """Test getting cache statistics."""
        cache = TaskCache(cache_dir=tmp_path / "cache")
        
        cache.put("fp1", {"data": "1"})
        cache.get("fp1")  # Hit
        cache.get("fp2")  # Miss
        
        stats = cache.get_stats()
        
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["total_requests"] == 2
        assert stats["hit_rate_pct"] == 50.0
        assert str(tmp_path / "cache") in stats["cache_dir"]
    
    def test_cache_path_subdirectory(self, tmp_path):
        """Test that cache uses subdirectories based on fingerprint prefix."""
        cache = TaskCache(cache_dir=tmp_path / "cache")
        
        fingerprint = "ab123456"
        cache.put(fingerprint, {"data": "test"})
        
        # Check that file is in subdirectory "ab"
        expected_path = tmp_path / "cache" / "ab" / f"{fingerprint}.pkl"
        assert expected_path.exists()
    
    def test_corrupted_cache_handled(self, tmp_path):
        """Test that corrupted cache files are handled gracefully."""
        cache = TaskCache(cache_dir=tmp_path / "cache")
        
        fingerprint = "corrupted"
        cache_file = cache._cache_path(fingerprint)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write invalid data
        cache_file.write_text("not a pickle")
        
        # Should return None and count as miss
        result = cache.get(fingerprint)
        assert result is None
        assert cache.misses == 1
