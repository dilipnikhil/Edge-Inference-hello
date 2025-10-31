"""
Advanced model optimization and export for edge inference
Includes ONNX optimization, ExecuTorch, TorchScript, quantization
"""

import torch
import torch.onnx
import torch.quantization
import numpy as np
import os
import argparse
from pathlib import Path
from model import TinyKWS, MinimalKWS


class ModelOptimizer:
    """Advanced model optimization for edge deployment"""
    
    def __init__(self, model_path, model_type="tiny"):
        self.model_path = model_path
        self.model_type = model_type
        self.device = "cpu"  # Always use CPU for quantization
        
        # Load model
        print(f"Loading model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if "model_type" in checkpoint:
            self.model_type = checkpoint["model_type"].replace("_quantized", "")
        
        if self.model_type == "tiny" or "tiny" in self.model_type:
            self.model = TinyKWS(num_classes=2, num_filters=64)
        else:
            self.model = MinimalKWS(num_classes=2, num_filters=32)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        print(f"Loaded {self.model_type} model")
    
    def export_onnx_optimized(self, output_path, opset_version=14, optimize=True):
        """
        Export to ONNX with optimizations
        Uses graph optimizations for better performance
        """
        print(f"\n{'='*60}")
        print("Exporting optimized ONNX model...")
        print(f"{'='*60}")
        
        dummy_input = torch.randn(1, 1, 98)
        
        # Export with optimizations
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["audio_features"],
            output_names=["logits"],
            dynamic_axes={"audio_features": {0: "batch_size"}, "logits": {0: "batch_size"}},
            verbose=False,
            training=torch.onnx.TrainingMode.EVAL
        )
        
        size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"✓ ONNX model exported: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
        
        # Apply ONNX Runtime optimizations if available
        if optimize:
            try:
                import onnx
                try:
                    from onnxruntime.transformers import optimizer as ort_optimizer
                except ImportError:
                    # Try alternative ONNX optimization
                    import onnxoptimizer
                
                print("\nApplying ONNX Runtime optimizations...")
                model = onnx.load(output_path)
                
                # Optimize model using onnxoptimizer
                model_graph = onnx.load(output_path)
                # Apply standard optimizations
                optimized_model = onnxoptimizer.optimize(model_graph, ['eliminate_nop_transpose',
                                                                         'fuse_bn_into_conv',
                                                                         'fuse_matmul_add_bias_into_gemm',
                                                                         'eliminate_nop_pad',
                                                                         'eliminate_unused_initializer'])
                
                optimized_path = output_path.replace(".onnx", "_optimized.onnx")
                onnx.save(optimized_model, optimized_path)
                
                optimized_size = os.path.getsize(optimized_path) / (1024**2)
                print(f"✓ Optimized ONNX model: {optimized_path}")
                print(f"  Size: {optimized_size:.2f} MB")
                
                return optimized_path
            except ImportError:
                print("  (ONNX Runtime optimizations not available)")
            except Exception as e:
                print(f"  (Optimization failed: {e})")
        
        return output_path
    
    def export_torchscript(self, output_path, optimize=True):
        """
        Export to TorchScript for optimized inference
        """
        print(f"\n{'='*60}")
        print("Exporting TorchScript model...")
        print(f"{'='*60}")
        
        dummy_input = torch.randn(1, 1, 98)
        
        # Trace model
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        if optimize:
            # Optimize traced model
            traced_model = torch.jit.optimize_for_inference(traced_model)
            print("✓ Applied TorchScript optimizations")
        
        # Save
        traced_model.save(output_path)
        
        size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"✓ TorchScript model exported: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
        
        return output_path
    
    def quantize_int8_dynamic(self, output_path):
        """
        Dynamic INT8 quantization (fast, no calibration needed)
        """
        print(f"\n{'='*60}")
        print("Quantizing to INT8 (Dynamic)...")
        print(f"{'='*60}")
        
        # Dynamic quantization
        quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            {torch.nn.Conv1d, torch.nn.Linear, torch.nn.BatchNorm1d},
            dtype=torch.qint8
        )
        
        # Save
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'model_type': self.model_type + "_int8_dynamic",
        }, output_path)
        
        size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"✓ INT8 Dynamic quantized model: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
        
        return output_path
    
    def quantize_int8_static(self, output_path, calibration_data=None):
        """
        Static INT8 quantization (better accuracy, requires calibration)
        
        Args:
            calibration_data: List of input tensors for calibration
        """
        print(f"\n{'='*60}")
        print("Quantizing to INT8 (Static)...")
        print(f"{'='*60}")
        
        if calibration_data is None:
            # Generate dummy calibration data
            print("Generating dummy calibration data...")
            calibration_data = [torch.randn(1, 1, 98) for _ in range(100)]
        
        # Prepare model for quantization
        self.model.eval()
        self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        
        # Prepare
        prepared_model = torch.quantization.prepare(self.model)
        
        # Calibrate
        print("Calibrating model...")
        with torch.no_grad():
            for data in calibration_data:
                prepared_model(data)
        
        # Convert
        quantized_model = torch.quantization.convert(prepared_model)
        
        # Save
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'model_type': self.model_type + "_int8_static",
        }, output_path)
        
        size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"✓ INT8 Static quantized model: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
        
        return output_path
    
    def export_executorch(self, output_path):
        """
        Export to ExecuTorch format for edge deployment
        Requires executorch package
        """
        print(f"\n{'='*60}")
        print("Exporting to ExecuTorch...")
        print(f"{'='*60}")
        
        try:
            from executorch.exir import to_edge
            from executorch.extension.pybindings.portable_lib import _load_for_executorch
            
            # Convert to Edge format
            edge_program = to_edge(torch.jit.script(self.model))
            
            # Export
            edge_program.export().save(output_path)
            
            size_mb = os.path.getsize(output_path) / (1024**2)
            print(f"✓ ExecuTorch model exported: {output_path}")
            print(f"  Size: {size_mb:.2f} MB")
            
            return output_path
            
        except ImportError:
            print("✗ ExecuTorch not installed")
            print("  Install with: pip install executorch")
            return None
        except Exception as e:
            print(f"✗ ExecuTorch export failed: {e}")
            return None
    
    def prune_model(self, output_path, pruning_ratio=0.3):
        """
        Apply magnitude-based pruning to reduce model size
        """
        print(f"\n{'='*60}")
        print(f"Pruning model ({pruning_ratio*100:.0f}% sparsity)...")
        print(f"{'='*60}")
        
        # Apply unstructured pruning
        import torch.nn.utils.prune as prune
        
        parameters_to_prune = []
        for module in self.model.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
                parameters_to_prune.append((module, 'weight'))
        
        # Prune
        for module, param_name in parameters_to_prune:
            prune.ln_structured(
                module,
                name='weight',
                amount=pruning_ratio,
                n=2,
                dim=0
            )
        
        # Make pruning permanent
        for module, _ in parameters_to_prune:
            prune.remove(module, 'weight')
        
        # Save pruned model
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_type': self.model_type + f"_pruned_{int(pruning_ratio*100)}",
        }, output_path)
        
        size_mb = os.path.getsize(output_path) / (1024**2)
        print(f"✓ Pruned model saved: {output_path}")
        print(f"  Size: {size_mb:.2f} MB")
        
        return output_path
    
    def benchmark_model(self, num_iterations=100):
        """Benchmark model inference speed"""
        print(f"\n{'='*60}")
        print("Benchmarking model...")
        print(f"{'='*60}")
        
        dummy_input = torch.randn(1, 1, 98)
        self.model.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = self.model(dummy_input)
        
        # Benchmark
        import time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = self.model(dummy_input)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - start_time
        
        avg_latency_ms = (elapsed / num_iterations) * 1000
        print(f"✓ Average latency: {avg_latency_ms:.2f} ms")
        print(f"  Throughput: {num_iterations/elapsed:.1f} inferences/sec")
        
        return avg_latency_ms


def main():
    parser = argparse.ArgumentParser(description="Advanced model optimization and export")
    parser.add_argument("--model", type=str, required=True, help="Path to PyTorch model")
    parser.add_argument("--model_type", type=str, default="tiny", choices=["tiny", "minimal"])
    parser.add_argument("--output_dir", type=str, default="exports_optimized", help="Output directory")
    parser.add_argument("--formats", type=str, nargs="+", 
                       choices=["onnx", "torchscript", "int8_dynamic", "int8_static", 
                               "executorch", "pruned", "all"],
                       default=["onnx", "int8_dynamic"],
                       help="Export formats")
    parser.add_argument("--pruning_ratio", type=float, default=0.3, help="Pruning ratio (0-1)")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    optimizer = ModelOptimizer(args.model, args.model_type)
    
    if args.benchmark:
        optimizer.benchmark_model()
    
    results = {}
    
    if "all" in args.formats:
        args.formats = ["onnx", "torchscript", "int8_dynamic", "executorch"]
    
    if "onnx" in args.formats:
        onnx_path = os.path.join(args.output_dir, "model_optimized.onnx")
        results["onnx"] = optimizer.export_onnx_optimized(onnx_path, optimize=True)
    
    if "torchscript" in args.formats:
        ts_path = os.path.join(args.output_dir, "model.torchscript")
        results["torchscript"] = optimizer.export_torchscript(ts_path, optimize=True)
    
    if "int8_dynamic" in args.formats:
        int8_path = os.path.join(args.output_dir, "model_int8_dynamic.pth")
        results["int8_dynamic"] = optimizer.quantize_int8_dynamic(int8_path)
    
    if "int8_static" in args.formats:
        int8_static_path = os.path.join(args.output_dir, "model_int8_static.pth")
        results["int8_static"] = optimizer.quantize_int8_static(int8_static_path)
    
    if "executorch" in args.formats:
        et_path = os.path.join(args.output_dir, "model.pte")
        results["executorch"] = optimizer.export_executorch(et_path)
    
    if "pruned" in args.formats:
        pruned_path = os.path.join(args.output_dir, f"model_pruned_{int(args.pruning_ratio*100)}.pth")
        results["pruned"] = optimizer.prune_model(pruned_path, args.pruning_ratio)
    
    print(f"\n{'='*60}")
    print("Export Summary:")
    print(f"{'='*60}")
    for format_name, path in results.items():
        if path:
            size = os.path.getsize(path) / (1024**2)
            print(f"  {format_name:15s}: {Path(path).name} ({size:.2f} MB)")
    
    print(f"\nAll exports saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

