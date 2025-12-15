#!/usr/bin/env python3
"""Unit tests for main.py"""

import subprocess
import sys
from unittest.mock import patch, MagicMock
import pytest

from main import get_gpu_info, find_free_gpus, format_output, main


class TestGetGpuInfo:
    """Test cases for get_gpu_info function."""

    def test_get_gpu_info_success(self):
        """Test successful nvidia-smi query parsing."""
        mock_output = """0, NVIDIA GeForce RTX 3080, 100, 10240
1, NVIDIA GeForce RTX 3080, 50, 10240
2, NVIDIA GeForce RTX 3080, 8000, 10240"""

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_output,
                stderr="",
                returncode=0
            )

            result = get_gpu_info()

            assert len(result) == 3
            assert result[0]["index"] == 0
            assert result[0]["name"] == "NVIDIA GeForce RTX 3080"
            assert result[0]["memory_used_mb"] == 100
            assert result[0]["memory_total_mb"] == 10240
            assert result[1]["memory_used_mb"] == 50
            assert result[2]["memory_used_mb"] == 8000

    def test_get_gpu_info_empty_output(self):
        """Test handling empty nvidia-smi output."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout="\n",
                stderr="",
                returncode=0
            )

            result = get_gpu_info()
            assert result == []

    def test_get_gpu_info_malformed_output(self):
        """Test handling malformed nvidia-smi output."""
        mock_output = """0, NVIDIA GeForce RTX 3080, 100
1, NVIDIA GeForce RTX 3080, 50, 10240
2, NVIDIA GeForce RTX 3080"""

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                stdout=mock_output,
                stderr="",
                returncode=0
            )

            result = get_gpu_info()
            # Only the well-formed line should be parsed
            assert len(result) == 1
            assert result[0]["index"] == 1
            assert result[0]["memory_used_mb"] == 50
            assert result[0]["memory_total_mb"] == 10240

    def test_get_gpu_info_subprocess_error(self):
        """Test handling subprocess.CalledProcessError."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                1, "nvidia-smi", stderr="Driver not loaded"
            )

            with patch('sys.exit') as mock_exit:
                get_gpu_info()
                mock_exit.assert_called_once_with(1)

    def test_get_gpu_info_file_not_found(self):
        """Test handling when nvidia-smi is not found."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = FileNotFoundError()

            with patch('sys.exit') as mock_exit:
                get_gpu_info()
                mock_exit.assert_called_once_with(1)

    def test_get_gpu_info_unexpected_error(self):
        """Test handling unexpected errors."""
        with patch('subprocess.run') as mock_run:
            mock_run.side_effect = Exception("Unexpected error")

            with patch('sys.exit') as mock_exit:
                get_gpu_info()
                mock_exit.assert_called_once_with(1)


class TestFindFreeGpus:
    """Test cases for find_free_gpus function."""

    def test_find_free_gpus_default_threshold(self):
        """Test finding free GPUs with default threshold (300MB)."""
        gpu_info = [
            {"index": 0, "name": "RTX 3080", "memory_used_mb": 100, "memory_total_mb": 10240},
            {"index": 1, "name": "RTX 3080", "memory_used_mb": 400, "memory_total_mb": 10240},
            {"index": 2, "name": "RTX 3080", "memory_used_mb": 50, "memory_total_mb": 10240},
        ]

        result = find_free_gpus(gpu_info)

        assert len(result) == 2
        assert result[0]["index"] == 0
        assert result[1]["index"] == 2

    def test_find_free_gpus_custom_threshold(self):
        """Test finding free GPUs with custom threshold."""
        gpu_info = [
            {"index": 0, "name": "RTX 3080", "memory_used_mb": 100, "memory_total_mb": 10240},
            {"index": 1, "name": "RTX 3080", "memory_used_mb": 400, "memory_total_mb": 10240},
            {"index": 2, "name": "RTX 3080", "memory_used_mb": 600, "memory_total_mb": 10240},
        ]

        result = find_free_gpus(gpu_info, threshold_mb=500)

        assert len(result) == 2
        assert result[0]["index"] == 0
        assert result[1]["index"] == 1

    def test_find_free_gpus_none_found(self):
        """Test when no GPUs are free."""
        gpu_info = [
            {"index": 0, "name": "RTX 3080", "memory_used_mb": 500, "memory_total_mb": 10240},
            {"index": 1, "name": "RTX 3080", "memory_used_mb": 800, "memory_total_mb": 10240},
        ]

        result = find_free_gpus(gpu_info)
        assert result == []

    def test_find_free_gpus_all_free(self):
        """Test when all GPUs are free."""
        gpu_info = [
            {"index": 0, "name": "RTX 3080", "memory_used_mb": 0, "memory_total_mb": 10240},
            {"index": 1, "name": "RTX 3080", "memory_used_mb": 200, "memory_total_mb": 10240},
        ]

        result = find_free_gpus(gpu_info)
        assert len(result) == 2

    def test_find_free_gpus_empty_input(self):
        """Test with empty GPU list."""
        result = find_free_gpus([])
        assert result == []


class TestFormatOutput:
    """Test cases for format_output function."""

    def test_format_output_no_gpus(self):
        """Test formatting when no free GPUs found."""
        result = format_output([])
        assert result == "No free GPUs found."

    def test_format_output_verbose(self):
        """Test verbose output formatting."""
        free_gpus = [
            {"index": 0, "name": "RTX 3080", "memory_used_mb": 100, "memory_total_mb": 10240},
            {"index": 2, "name": "RTX 3080", "memory_used_mb": 50, "memory_total_mb": 10240},
        ]

        result = format_output(free_gpus, verbose=True)

        assert "Free GPUs found:" in result
        assert "GPU 0: RTX 3080 (100MB / 10240MB used)" in result
        assert "GPU 2: RTX 3080 (50MB / 10240MB used)" in result

    def test_format_output_simple(self):
        """Test simple output formatting (just indexes)."""
        free_gpus = [
            {"index": 0, "name": "RTX 3080", "memory_used_mb": 100, "memory_total_mb": 10240},
            {"index": 2, "name": "RTX 3080", "memory_used_mb": 50, "memory_total_mb": 10240},
        ]

        result = format_output(free_gpus, verbose=False)
        assert result == "0 2"

    def test_format_output_single_gpu(self):
        """Test formatting with single GPU."""
        free_gpus = [
            {"index": 1, "name": "RTX 3080", "memory_used_mb": 100, "memory_total_mb": 10240}
        ]

        result = format_output(free_gpus, verbose=False)
        assert result == "1"

        result_verbose = format_output(free_gpus, verbose=True)
        assert "GPU 1: RTX 3080 (100MB / 10240MB used)" in result_verbose


class TestMain:
    """Test cases for main function."""

    @patch('main.print')
    @patch('main.get_gpu_info')
    @patch('sys.argv', ['main.py'])
    def test_main_default_output(self, mock_get_gpu_info, mock_print):
        """Test main with default arguments."""
        mock_get_gpu_info.return_value = [
            {"index": 0, "name": "RTX 3080", "memory_used_mb": 100, "memory_total_mb": 10240},
            {"index": 1, "name": "RTX 3080", "memory_used_mb": 400, "memory_total_mb": 10240},
        ]

        main()

        # Should print the GPU indexes
        mock_print.assert_called_with("0")

    @patch('main.print')
    @patch('main.get_gpu_info')
    @patch('sys.argv', ['main.py', '-v'])
    def test_main_verbose_output(self, mock_get_gpu_info, mock_print):
        """Test main with verbose output."""
        mock_get_gpu_info.return_value = [
            {"index": 0, "name": "RTX 3080", "memory_used_mb": 100, "memory_total_mb": 10240},
        ]

        main()

        # Should print verbose output
        mock_print.assert_called()
        args, _ = mock_print.call_args
        assert "Free GPUs found:" in args[0]
        assert "GPU 0: RTX 3080 (100MB / 10240MB used)" in args[0]

    @patch('main.print')
    @patch('main.get_gpu_info')
    @patch('sys.argv', ['main.py', '-t', '500'])
    def test_main_custom_threshold(self, mock_get_gpu_info, mock_print):
        """Test main with custom threshold."""
        mock_get_gpu_info.return_value = [
            {"index": 0, "name": "RTX 3080", "memory_used_mb": 400, "memory_total_mb": 10240},
            {"index": 1, "name": "RTX 3080", "memory_used_mb": 600, "memory_total_mb": 10240},
        ]

        main()

        # Should include GPU 0 with 400MB usage (threshold is 500MB)
        mock_print.assert_called_with("0")

    @patch('main.print')
    @patch('main.get_gpu_info')
    @patch('sys.argv', ['main.py', '-q'])
    def test_main_quiet_mode_no_gpus(self, mock_get_gpu_info, mock_print):
        """Test main in quiet mode with no free GPUs."""
        mock_get_gpu_info.return_value = [
            {"index": 0, "name": "RTX 3080", "memory_used_mb": 500, "memory_total_mb": 10240},
        ]

        main()

        # Should not print anything in quiet mode
        mock_print.assert_not_called()

    @patch('main.print')
    @patch('main.get_gpu_info')
    @patch('sys.argv', ['main.py'])
    def test_main_no_gpus_found(self, mock_get_gpu_info, mock_print):
        """Test main when no GPUs are detected."""
        mock_get_gpu_info.return_value = []

        with patch('sys.exit') as mock_exit:
            main()
            mock_exit.assert_called_once_with(1)

    @patch('main.print')
    @patch('main.sys.exit')
    @patch('main.get_gpu_info')
    @patch('sys.argv', ['main.py'])
    def test_main_no_gpus_quiet(self, mock_get_gpu_info, mock_exit, mock_print):
        """Test main with no free GPUs (not quiet mode)."""
        mock_get_gpu_info.return_value = [
            {"index": 0, "name": "RTX 3080", "memory_used_mb": 500, "memory_total_mb": 10240},
        ]

        main()

        # Should print message when not in quiet mode
        mock_print.assert_called_with("No free GPUs found.")