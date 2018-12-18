function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

  % Calculate 'optimal' C and sigma
  % Cvals =     [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
  % sigmavals = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
  %
  % minError = realmax;
  % minC = 0;
  % minSigma = 0;
  %
  % for ci = 1:size(Cvals)
  %   for si = 1:size(sigmavals)
  %     cc = Cvals(ci);
  %     sc = sigmavals(si);
  %     model = svmTrain(X, y, cc, @(x1, x2) gaussianKernel(x1, x2, sc));
  %     predictions = svmPredict(model, Xval);
  %     errors = mean(double(predictions ~= yval));
  %
  %     if errors < minError
  %       minError = errors;
  %       minC = Cvals(ci);
  %       minSigma = sigmavals(si);
  %     end
  %   end
  % end
  %
  % C = minC
  % sigma = minSigma

  % Best results from above
  C = 1;
  sigma = 0.1;
end
