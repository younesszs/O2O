

        data {
          int<lower=1> M;                  // number of gangs
          array[M] int<lower=1> Na;        // list of lenght for each on_data
          array[M] int<lower=1> Nb;        // list of lenght for each off_data
          int<lower=1> maxNa;              // maximum over Na
          int<lower=1> maxNb;              // maximum over Nb
          array[M] vector[maxNa] ta;
          array[M] vector[maxNb] tb;
          int<lower=0> T;
        }
        parameters {
          matrix<lower=0>[M,2] mu;                     // baseline
          matrix<lower=0>[2,2] gamma;                  // decay
          matrix<lower=0, upper=1>[2,2] alpha;         // adjacency
        }
        transformed parameters {
          array[M] vector[maxNa] lam_a;
          array[M] vector[maxNb] lam_b;

          // initialize first elements
          for (m in 1:M) {
            lam_a[m][1] = mu[m,1];
            lam_b[m][1] = mu[m,2];
          }

          // lam online
          for (m in 1:M) {
            for (j in 1:Na[m]) {
              lam_a[m][j] = mu[m,1];
              for (k in 1:(j-1)) {
                if (ta[m][j] > ta[m][k] && ta[m][j] != -1 && ta[m][k] != -1) {
                  lam_a[m][j] += alpha[1,1] * gamma[1,1] * exp(-gamma[1,1] * (ta[m][j] - ta[m][k]));
                }
              }
              for (k in 1:Nb[m]) {
                if (ta[m][j] > tb[m][k] && ta[m][j] != -1 && tb[m][k] != -1) {
                  lam_a[m][j] += alpha[1,2] * gamma[1,2] * exp(-gamma[1,2] * (ta[m][j] - tb[m][k]));
                }
              }
            }

            // lam offline
            for (j in 1:Nb[m]) {
              lam_b[m][j] = mu[m,2];
              for (k in 1:(j-1)) {
                if (tb[m][j] > tb[m][k] && tb[m][j] != -1 && tb[m][k] != -1) {
                  lam_b[m][j] += alpha[2,2] * gamma[2,2] * exp(-gamma[2,2] * (tb[m][j] - tb[m][k]));
                }
              }
              for (k in 1:Na[m]) {
                if (tb[m][j] > ta[m][k] && tb[m][j] != -1 && ta[m][k] != -1) {
                  lam_b[m][j] += alpha[2,1] * gamma[2,1] * exp(-gamma[2,1] * (tb[m][j] - ta[m][k]));
                }
              }
            }
          }
        }

        model {
          // priors
          alpha[1,1] ~ beta(1,1);
          alpha[1,2] ~ beta(1,1);
          alpha[2,1] ~ beta(1,1);
          alpha[2,2] ~ beta(1,1);

          for (m in 1:M) {
            mu[m,1] ~ cauchy(0,5);
            mu[m,2] ~ cauchy(0,5);
          }
          gamma[1,1] ~ cauchy(0,5);
          gamma[2,1] ~ cauchy(0,5);
          gamma[1,2] ~ cauchy(0,5);
          gamma[2,2] ~ cauchy(0,5);

          // likelihood maximization using the Shoenberg approximation
          for (m in 1:M) {
            for (j in 1:Na[m]) {
              target += log(lam_a[m][j]);
            }
            for (j in 1:Nb[m]) {
              target += log(lam_b[m][j]);
            }
            target += -mu[m,1] * T -mu[m,2] * T - (alpha[1,1] + alpha[2,1]) * Na[m] - (alpha[1,2] + alpha[2,2]) * Nb[m];
          }
        }

