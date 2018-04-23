
%Data simulation:
n=1000; %Try 1000 and 10000

%Exogenous regressor simulation:
z1 = randn(n,1)+1; %Education in years, standardized around 1
%note that 1960s/70s HS dropout and college attendance rates were both ~30-40%
z2 = randn(n,1)+1;  %work experience in years, standardized around 1

%Instrumental variable: draft lottery number
x1 = randn(n,1)+1;
%uniformly dist. integers would be more realistic but for simulating 
%corr(x1,vet) this will work as well

%Normally distributed error terms
u = randn(n,1); %other determinants of income
v = randn(n,1); %other determinants of why someone joins the military

%Simulate the endogenous regressor, (veteran status). Is negatively correlated
%with lottery number, also negatively correlated somewhat with
%education and years of civilian work experience (less years spent in the US)
%note slide 183 "any regressor orthogonal to u can be an instrument" 
%z1 z2 are orthogonal to u in our simulation but in reality, not certain
%Thus we make vet_raw dependent on z1,z2,x1, and must depend partially on 
%u (the source of endogeneity), and other factors affecting
%why someone joins the military (v). 
vet_raw = - 0.75*z1 - 0.75*z2 - 1*x1 + 0.5*u + 0.75*v;
mean_vetraw = mean(vet_raw);

%Check to make sure it's realistic - z1, z2, x1 should all be significant but 
%there should still be unexplained variance due to u and v
%X_vetpred = [z1 z2 x1];
%regstats(vet_raw,X_vetpred,'linear') %R^2 = 0.73


%Turn veteran into a categorical variable (0=not veteran, 1=veteran)
%Choose a cut-off of vet_raw such that ~20% of people served in Vietnam
vet = vet_raw>-1.13; 
mean_vet = mean(vet);
corr_x1_vet = corr(x1,vet); %~-0.4
%%
%Response variable, hourly wage
%Simulated as depending on all the regressors
y = 5 + 0.75*z1 + 0.75*z2 - 2*vet + u;
%min(y)
disp('mean income')
mean(y) %~$6  (close to the average unadjusted wage for the 1980s)
%max(y)
histogram(y); 
var(y) %~3.3

%% Estimation by OLS - we expect to see biased coefficient estimates
X = [ones(size(z1)) z1 z2 vet];
b = regress(y,X)   
%increasing n to 10,000 has negligible difference

% Additional regression results
%X = [z1 z2 vet];
%regstats(y,X,'linear')
%select t-statistics and coefficient covariance (to get covb) 
% Standard errors
%stderrOLS  = sqrt(diag(covb)); 

%R-square is 0.73 but comparison to 2SLS won't be meaningful here:
%https://www.stata.com/support/faqs/statistics/two-stage-least-squares/

%% Can two stage least squares help?

Z = [ones(size(z1)) z1 z2 vet]; %all the regressors
X = [ones(size(z1)) x1 z1 z2]; %all the instruments

% # obs
n;
% # of regressors in the original model
k = 3; 
% Degrees of freedom
df = n - k; 

% Stage 1 of 2SLS: Regress regressors in Z on instruments in X, 
% and get fitted values for endogenous vet
P = X*inv(X'*X)*X'; %projection matrix, slide 202
Z_hat = P*Z;

% Stage 2: Regress y on fitted values  slide 203
beta_2sls = inv(Z'*P'*P*Z)*Z'*P*y;
coef_2sls      = beta_2sls

%Increasing n to 10,000 brings coefficients to near perfect --> consistent

%residuals
u_iv = y - Z*beta_2sls;

%estimation of variance (slide 87)
s_hat   = u_iv'*u_iv/df; 

% Estimated covariance matrix 
%https://www.fsb.miamioh.edu/lij14/411_note_2sls.pdf  pg.9
var_hat = s_hat*(inv(Z'*P*Z)); 

% Standard errors
stderr_2sls  = sqrt(diag(var_hat)); 

% t-ratios
t_stat  = beta_2sls./stderr;

% p-values
pval_2sls = betainc(df./(df+(1.*t_stat.^2)),(df./2),(1./2));


%% Weak instruments
%% Simulate x1 as a weak instrument (not highly correlated with vet)
%Data simulation:
n=1000; %Try 1000 and 10000

%Exogenous regressor simulation:
z1 = randn(n,1)+1; %Education in years, standardized around 1
z2 = randn(n,1)+1;  %work experience in years, standardized around 1

%Instrumental variable: draft lottery number
x1 = randn(n,1)+1;

%Normally distributed error terms
u = randn(n,1); %other determinants of income
v = randn(n,1); %other determinants of why someone joins the military

vet_raw = - 0.75*z1 - 0.75*z2 - 0.2*x1 + 0.5*u + 0.75*v;
mean_vetraw = mean(vet_raw);

%Choose a cut-off of vet_raw such that ~20% of people served in Vietnam
vet = vet_raw>-0.4; 
mean_vet = mean(vet);

corr_x1_vet = corr(vet, x1); %correlation reduced from -0.4 to ~-0.10 (varies a lot)

%Response variable, hourly wage
y = 5 + 0.75*z1 + 0.75*z2 - 2*vet + u;

%% Estimation by OLS - we expect to see biased coefficient estimates
X = [ones(size(z1)) z1 z2 vet];
b = regress(y,X)   
X = [z1 z2 vet];
regstats(y,X,'linear')
% Standard errors
stderrOLS  = sqrt(diag(covb)); 
%% 2SLS - with the lottery numbers having such a small impact, we again expect bias
Z = [ones(size(z1)) z1 z2 vet]; %all the regressors
X = [ones(size(z1)) x1 z1 z2]; %all the instruments

k = 3; 
df = n - k; 
% Stage 1 of 2SLS: Regress regressors in Z on instruments in X, 
% and get fitted values for endogenous vet
P = X*inv(X'*X)*X'; %projection matrix, slide 202
Z_hat = P*Z;

% Stage 2: Regress y on fitted values  slide 203
beta_2sls = inv(Z'*P'*P*Z)*Z'*P*y;
coef_2sls      = beta_2sls

%residuals
u_iv = y - Z*beta_2sls;

%estimation of variance (slide 87)
s_hat   = u_iv'*u_iv/df; 

% Estimated covariance matrix 
%https://www.fsb.miamioh.edu/lij14/411_note_2sls.pdf  pg.9
var_hat = s_hat*(inv(Z'*P*Z)); 

% Standard errors
stderr_2sls  = sqrt(diag(var_hat)); 

% t-ratios
t_stat  = beta_2sls./stderr;

% p-values
pval_2sls = betainc(df./(df+(1.*t_stat.^2)),(df./2),(1./2));





