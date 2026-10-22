inline void pack(std::vector<int>& p, const CudaOperationGeometry::Resize& g){pack(p,g.c);pack(p,g.ih);pack(p,g.iw);pack(p,g.oh);pack(p,g.ow);pack(p,g.sh);pack(p,g.sw);}
inline void pack(std::vector<int>& p, const CudaOperationGeometry::Pool& g){pack(p,g.c);pack(p,g.h);pack(p,g.w);pack(p,g.kh);pack(p,g.kw);pack(p,g.sh);pack(p,g.sw);pack(p,g.ph);pack(p,g.pw);pack(p,g.oh);pack(p,g.ow);pack(p,g.maximum);pack(p,g.token);}
inline void pack(std::vector<int>& p, const CudaOperationGeometry::Projection& g){pack(p,g.mode);pack(p,g.dim);pack(p,g.vocab);pack(p,g.pad);pack(p,g.rows);pack(p,g.ids);pack(p,g.features);pack(p,g.second);pack(p,g.prefix);pack(p,g.output);pack(p,g.parameters);pack(p,g.bias);}
inline void pack(std::vector<int>& p, const CudaOperationGeometry::ExtendedConv& g){pack(p,g.mode);pack(p,g.ic);pack(p,g.oc);pack(p,g.h);pack(p,g.w);pack(p,g.kh);pack(p,g.kw);pack(p,g.stride);pack(p,g.pad);pack(p,g.oh);pack(p,g.ow);pack(p,g.parameters);pack(p,g.bias);}
inline void pack(std::vector<int>& p, const CudaOperationGeometry::BatchNorm& g){pack(p,g.channels);pack(p,g.spatial);pack(p,g.batch);pack(p,g.affine);pack(p,g.bias);pack(p,g.running);}
inline void pack(std::vector<int>& p, const CudaOperationGeometry::Attention& g){pack(p,g.ed);pack(p,g.ked);pack(p,g.heads);pack(p,g.q);pack(p,g.k);pack(p,g.cross);pack(p,g.causal);pack(p,g.qbias);pack(p,g.obias);pack(p,g.mainParameters);pack(p,g.output);}
inline void pack(std::vector<int>& p, const CudaOperationGeometry::Recurrent& g){pack(p,g.steps);pack(p,g.input);pack(p,g.hidden);pack(p,g.gates);pack(p,g.parameters);pack(p,g.workspace);}
inline int kernelId(const std::string& name){
if(name=="norm_forward")return 0;
if(name=="norm_backward")return 1;
if(name=="channel_gradient")return 2;
if(name=="resize_forward")return 3;
if(name=="nearest_backward")return 4;
if(name=="bilinear_backward")return 5;
if(name=="causal_mask")return 6;
if(name=="softmax_kernel")return 7;
if(name=="im2col")return 8;
if(name=="col2im")return 9;
if(name=="row_norm_kernel")return 10;
if(name=="row_norm_grad")return 11;
if(name=="projection_kernel")return 12;
if(name=="geglu_kernel")return 13;
if(name=="bicubic_kernel")return 14;
if(name=="pool_kernel")return 15;
if(name=="permute_kernel")return 16;
if(name=="spatial_move")return 17;
if(name=="extended_conv_kernel")return 18;
if(name=="dropout_kernel")return 19;
if(name=="reparameterize_kernel")return 20;
if(name=="batch_norm_kernel")return 21;
if(name=="attention_add_bias")return 22;
if(name=="attention_project")return 23;
if(name=="attention_probabilities")return 24;
if(name=="attention_context")return 25;
if(name=="attention_output")return 26;
if(name=="attention_score_grad")return 27;
if(name=="attention_projected_grad")return 28;
if(name=="attention_weight_grad")return 29;
if(name=="attention_input_grad")return 30;
if(name=="recurrent_kernel")return 31;
if(name=="nms_kernel")return 32;
if(name=="nms_stop_gradient_kernel")return 33;
if(name=="cuda_unary_kernel")return 34;
if(name=="cuda_binary_kernel")return 35;
if(name=="cuda_unary_backward_kernel")return 36;
if(name=="cuda_binary_backward_kernel")return 37;
if(name=="portable_copy")return 38;
if(name=="portable_gemm")return 39;
if(name=="portable_ger")return 40;
if(name=="portable_sequence_constant_forward")return 41;
if(name=="portable_sequence_constant_backward")return 42;
return -1;}
inline std::vector<int> kernelWriteArguments(int id){switch(id){
case 0:return {5,6,7};
case 1:return {7};
case 2:return {5,6};
case 3:return {4};
case 4:return {4};
case 5:return {4};
case 6:return {3};
case 7:return {5};
case 8:return {4};
case 9:return {4};
case 10:return {6,7};
case 11:return {5,6};
case 12:return {7,8,9,10};
case 13:return {5};
case 14:return {5};
case 15:return {5,6};
case 16:return {4};
case 17:return {4};
case 18:return {6,7,8};
case 19:return {5};
case 20:return {7,8};
case 21:return {7,8};
case 22:return {3};
case 23:return {5};
case 24:return {5};
case 25:return {5};
case 26:return {6};
case 27:return {6};
case 28:return {8};
case 29:return {10};
case 30:return {7};
case 31:return {3,4,5,6,7,8,9};
case 32:return {6,7,8};
case 33:return {3};
case 34:return {4};
case 35:return {5};
case 36:return {5};
case 37:return {6,7};
case 38:return {4};
case 39:return {5};
case 40:return {5};
case 41:return {5};
case 42:return {5};
default:return {};}}
