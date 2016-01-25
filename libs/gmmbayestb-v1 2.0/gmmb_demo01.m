% GMMB_DEMO01   Demostrate GMMBayes mixture learning and data classification.
%        This demo generates some Gaussian mixture distributed data,
%        divides it into training and test set, runs Figueiredo-Jain
%        algorithm on the training set and classifies the test set.
%
%
% References:
%
% Author(s):
%    Pekka Paalanen <pekka.paalanen@lut.fi>
%
% Copyright:
%
%   Bayesian Classifier with Gaussian Mixture Model Pdf
%   functionality is Copyright (C) 2003 by Pekka Paalanen and
%   Joni-Kristian Kamarainen.
%
%   $Name:  $ $Revision: 1.2 $  $Date: 2005/04/14 10:33:34 $

%function [] = gmmb_demo01

close all;clear all;clc;
disp('Generating data from three classes with 3, 1 and 2 Gaussian components...');

% generate test data
alldata = [ ...
	mvnrnd([2 1], covrot(1, 0.7, 1), 200) ;...
	mvnrnd([-2 1], covrot(0.4, 1.2, pi/3), 200) ;...
	mvnrnd([0 1.5], covrot(0.5, 0.5, 0), 150) ;...
	mvnrnd([-3 -1.5], covrot(0.5, 0.5, 0), 150) ;...
	mvnrnd([3 -1.5], covrot(0.5, 0.5, 0), 150) ;...
	mvnrnd([0 -2.5], covrot(2.5, 1.5, 0), 200) ;...
	];

alltype = [ ...
	1*ones(200,1); ...
	1*ones(200,1); ...
	2*ones(150,1); ...
	3*ones(150,1); ...
	3*ones(150,1); ...
	1*ones(200,1); ...
	];

disp('Separating test set (30%) and training set (70%)...');
[Ptrain Ttrain Ptest Ttest] = subset(alldata, alltype, round(size(alltype, 1)*0.70));

figH = figure;
plot_data(Ptrain, Ttrain, ['xr'; 'xb'; 'xg']);
disp('Now we have this kind of training set, three classes.');
disp('Next we will use the FJ algorithm to learn those classes.');
input('<press enter>');


FJ_params = { 'Cmax', 25, 'thr', 1e-3, 'animate', 1 }
disp('Running FJ...');

	bayesS = gmmb_create(Ptrain, Ttrain, 'FJ', FJ_params{:});

disp('Training complete.');
disp('There are now 3 more figures open, in those you can see how the FJ learned the distributions.');
input('<press enter>');



figure(figH);
disp('This is our test set. Let''s forget the class labels and classify the samples.');
plot_data(Ptest, Ttest, ['xr'; 'xb'; 'xg']);
input('<press enter>');


	% This is the Bayesian case.
	pdfmat = gmmb_pdf(Ptest, bayesS);
	postprob = gmmb_normalize( gmmb_weightprior(pdfmat, bayesS) );
	result = gmmb_decide(postprob);

disp('Done classifying. We used the Bayesian classifier.');

plot_data(Ptest, result, ['xr'; 'xb'; 'xg']);
rat = sum(result == Ttest) / length(Ttest);
disp(['We got ' num2str(rat*100) ' percent correct.']);
disp('The misclassified points are circled.');
miss = Ptest(result ~= Ttest, :);
hold on
plot(miss(:,1), miss(:,2), 'ok');
input('<press enter>');


figure

	% pdfmat and postprob are already computed
	histS = gmmb_generatehist(bayesS, 1000);
	outlier_mask = gmmb_fracthresh(pdfmat, histS, 0.9);
	postprob(outlier_mask) = 0;
	result = gmmb_decide(postprob);
	
	% Notice that in this case we chose:
	% a) for each point, discard the classes that do not
	%    pass the threshold and then choose the winner of the
	%    remaining classes, not vice versa.
	% b) not to normalize the posteriors after thresholding.

plot_data(Ptest, result+1, ['.k'; 'xr'; 'xb'; 'xg']);
miss = Ptest((result ~= Ttest)&(result~=0), :);
hold on
plot(miss(:,1), miss(:,2), 'ok');
disp('Here we classified the test data again using threshold of density quantile=0.9.');
disp('The points classified as outliers are black dots.');
disp('The misclassified points, that are not outliers, are circled.');
input('<press enter>');

disp('The End.');


