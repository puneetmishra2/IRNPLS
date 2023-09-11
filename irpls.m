function [crbvar,beta,weights,attributes] = irpls(X,Y,lvs,alpha)
dims = size(X);
if size(dims,2)>2
    X = reshape(X,dims(1),prod(dims(2:end)));
end
Xorig = X;  % saving a copy of X matrix for estimaiton of regression coefficient
yorig = Y;  % saving a copy of response to estimate the offset for regression coefficient
X2 = X-median(X,1); % saving a copy of median centered predictor
X = X-median(X,1); % median centering of predictor
Y = Y-median(Y,1); % median centering of response
q = zeros(size(Y,2),lvs); % pre-allocation for coefficient of Y with respect to T
T = zeros(size(Y,1),lvs); % pre-allocation for scores
crbvar = zeros(lvs,1); % pre-allocation for variation explained in Y
vartot = sum(sum(Y.*Y)); % total variation in response Y
weights = zeros(lvs,size(X,1)); % pre-allocation of weights
for i = 1:lvs  % loop for selecting variables
    D = eye(size(X,1))*(1/size(X,1));
    crit = 1;
    while crit>10e-5
        vt = X'*D*Y;
        if size(dims,2)>2
            vt = reshape(vt,dims(:,2:end));
            [wjx,~,wkx] = svds(vt,1);
            v = kron(wkx,wjx);
        else
            v = vt;
        end
        t_temp = X*v/norm(X*v);
        q_temp = (Y'*D)*t_temp;
        r = Y - t_temp*q_temp';
        lo = 1-diag(t_temp*t_temp');
        %lo = rescale(lo,0.01,1);
        r = r./sqrt(lo);
        r = r.*(0.6745/alpha)./mad(r,1);
        temp_D = (1-r.^2).^2;
        temp_D(abs(r)>=1) = 0;
        temp_D = prod(temp_D,2);
        crit = sum((abs(temp_D)-abs(diag(D))));
        D = diag(temp_D); 
    end
    weights(i,:) = diag(D);
    vt = X'*D*Y;
    if size(dims,2)>2
        vt = reshape(vt,dims(:,2:end));
        [wjx,~,wkx] = svds(vt,1);
        w(:,i) = kron(wkx,wjx);
    else
        w(:,i) = vt;  
    end
    t = X*w(:,i)/norm(X*w(:,i)); % scores based on loading weight
    t = t./norm(t); % Normalize the score 
    T(:,i) = t;
    q(:,i)= (Y'*D)*t; % Regression coeff wrt T(:,i)
    Pb(:,i) = (X'*D)*t; % Regression coeff wrt T(:,i)
    X = X - t*Pb(:,i)';
    Y = Y - t*q(:,i)'; % Calculate Y-residuals
    w(:,i) = w(:,i)/norm(w(:,i));
    crbvar(i,1) = (vartot - sum(Y(:).^2))/vartot; % Estimate explained variance 
end
%%%%%%%%%%%%%%%%%% post-processing for regression vector %%%%%%%%%%%%%%%%%%%%%%%%%
PtW = triu(Pb'*w); % The W-coordinates of (the projected) P
for i =1:size(Y,2) % The X-regression coefficients and intercepts for responses
    beta{i}  = cumsum(bsxfun(@times,w/PtW, q(i,:)),2);
    beta{i}  = [median(yorig(:,i)) - median(Xorig)*beta{i}; beta{i}];
end
attributes = struct('T', T, 'Pb', Pb,  'W', w, 'PtW', PtW);
end


