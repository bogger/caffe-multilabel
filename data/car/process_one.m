function [ds, bs] = process_one(im, model)
% Detect objects that score above a threshold.
%   [ds, bs] = process(im, model, thresh)
%
%   If the threshold is not included we use the one in the model.
%   This should lead to high-recall but low precision.
%
% Return values
%   ds        Clipped detection windows
%   bs        Boxes for all placed filters
%
% Arguments
%   im        Image
%   model     Object model
%   thresh    Detection threshold
thresh=0;
[ds, bs] = imgdetect(im, model, thresh);
if isempty(ds)
    error('empty detection!');
end

if ~isempty(ds)
  if model.type == model_types.MixStar
    if isfield(model, 'bboxpred')
      bboxpred = model.bboxpred;
      [ds, bs] = clipboxes(im, ds, bs);
      [ds, bs] = bboxpred_get(bboxpred, ds, reduceboxes(model, bs));
    else
      warning('no bounding box predictor found');
    end
  end
  [ds, bs] = clipboxes(im, ds, bs);
  %return the bbox with highest score
  
end
