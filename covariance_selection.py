# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as LA
import copy

import math
from scipy.stats import chi2

from graphviz import Graph # 無向グラフを作成するlibrary
from pprint import pprint
from PIL import Image

import pandas as pd
import json

pd.options.display.precision = 2 # 小数点以下2桁と設定
pd.options.display.float_format = '{:.4f}'.format # 有効数字4桁と設定

class CovarianceSelection:
    def __init__(self, data):
        self.data = data
        self.data_np = self.data
        self.rc_name = self.data.columns.values
        self.result = {'comment' : 'initial'}
        self.init_matrix()
        self.init_graph()
        self.I = []
        self.residual = np.zeros((self.data_np.shape[1], self.data_np.shape[1]))
        self.deviance = []

    # グラフの初期化
    def init_graph(self):
        g = Graph(format='png')

        for nnm in self.rc_name:
            g.node(nnm)

        self.graph_matrix = np.zeros((self.data_np.shape[1], self.data_np.shape[1]))
        for i in range(self.graph_matrix.shape[0]):
            for j in range(self.graph_matrix.shape[1]):
                if i > j:
                    self.graph_matrix[i][j] = 1
                    g.edge(self.rc_name[i], self.rc_name[j])
        
        file_name = './graph_init' # 保存する画像のファイル名を指定(拡張子(.png)を除く)
        g.render(filename=file_name, format='png', cleanup=True, directory=None)

        im = Image.open(file_name + '.png')
        im.show()

    # グラフを変更(辺を削除，復元)
    def change_graph(self, cnt, indices, restore_flg=False):
        g = Graph(format='png')
        
        for nnm in self.rc_name:
            g.node(nnm)
        
        if restore_flg:
            indices = sorted(indices, reverse=True)
            self.graph_matrix[indices[0]][indices[1]] = 1
            print('Graph Matrix : ',self.graph_matrix)
            for i in range(self.graph_matrix.shape[0]):
                for j in range(self.graph_matrix.shape[1]):
                    if i > j and self.graph_matrix[i][j] == 1:
                        g.edge(self.rc_name[i], self.rc_name[j])
            file_name = './graph_restored_' + str(cnt) # 保存する画像のファイル名を指定(拡張子(.png)を除く)
        else:
            indices = sorted(indices, reverse=True)
            self.graph_matrix[indices[0]][indices[1]] = 0
            print('Graph Matrix : ', self.graph_matrix)
            for i in range(self.graph_matrix.shape[0]):
                for j in range(self.graph_matrix.shape[1]):
                    if i > j and self.graph_matrix[i][j] == 1:
                        g.edge(self.rc_name[i], self.rc_name[j])
            file_name = './graph_' + str(cnt) # 保存する画像のファイル名を指定(拡張子(.png)を除く)
        g.render(filename=file_name, format='png', cleanup=True, directory=None)

        im = Image.open(file_name + '.png')
        im.show()

    # データから，標本分散共分散行列，標本相関行列，標本偏相関行列, を計算
    def init_matrix(self): 
        # variance-covariance matrix
        self.COV_mat = self.data.cov()
        print('> variance-covariance matrix')
        pprint(self.COV_mat)
        self.COV_mat = self.COV_mat.values
        self.i_COV_mat = copy.deepcopy(self.data.cov().values)
        print('\n')

        # correlation matrix
        print('> correlation matrix')
        self.COR_mat = self.data.corr()
        self.result['initial cor-mat'] = self.COR_mat.to_latex().replace('\n','')
        pprint((self.COR_mat))
        self.COR_mat = self.COR_mat.values
        self.i_COR_mat = copy.deepcopy(self.COR_mat)
        print('\n')

        # partial correlation matrix
        self.PCOR_mat = self.pcor_mat(self.COR_mat)
        self.result['initial pcor-mat'] = pd.DataFrame(self.pcor_mat(self.COR_mat), index=self.rc_name, columns=[self.rc_name]).to_latex().replace('\n','')
        print('> partial correlation matrix')
        pprint(pd.DataFrame(self.pcor_mat(self.PCOR_mat), index=self.rc_name, columns=[self.rc_name]))
        print('\n')

    # 相関行列を計算
    def cor_mat(self, S): # S : 標本分散共分散行列
        if LA.det(S)==0:
            return None
        size = S.shape[0]
        R = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                R[i][j] = S[i][j]/(math.sqrt(S[i][i])*math.sqrt(S[j][j]))
        
        return R

    # 偏相関行列
    def pcor_mat(self, R): # R : 標本相関行列
        if LA.det(R)==0:
            return None
        size = R.shape[0]
        R_inv = LA.inv(R)
        PCOR = np.zeros((size, size))

        for i in range(size):
            for j in range(size):
                if i==j:
                    PCOR[i][j] = 1
                else:
                    PCOR[i][j] = -R_inv[i][j]/(math.sqrt(R_inv[i][i])*math.sqrt(R_inv[j][j]))
        
        return PCOR

    # 制約条件下での分散共分散行列の推定量
    def e_covmat(self, indices, S):
        if LA.det(S)==0:
            return None
        S_inv = LA.inv(S)
        M = S
        M_inv = LA.inv(M)
        size = M_inv.shape[0]

        D = S_inv[indices[0]][indices[0]]*S_inv[indices[1]][indices[1]] - (S_inv[indices[0]][indices[1]])**2
        M_inv[indices[0]][indices[1]] = 0
        M_inv[indices[1]][indices[0]] = 0
        M_inv[indices[0]][indices[0]] = D / S_inv[indices[1]][indices[1]]
        M_inv[indices[1],indices[1]] = D / S_inv[indices[0]][indices[0]]
        
        for k in range(size):
            if k != indices[0] and k != indices[1]:
                M_inv[indices[0]][k] = S_inv[indices[0]][k] - S_inv[indices[0]][indices[1]]*S_inv[indices[1]][k]/S_inv[indices[1]][indices[1]]
                M_inv[indices[1]][k] = S_inv[indices[1]][k] - S_inv[indices[0]][indices[1]]*S_inv[indices[0]][k]/S_inv[indices[0]][indices[0]]

        for k in range(size):
            for l in range(size):
                if k != indices[0] and k != indices[1]:
                    M_inv[k][l] = S_inv[k][l] - (S_inv[indices[0]][indices[1]]/D)*(S_inv[indices[0]][k]*(S_inv[indices[1]][l] - S_inv[indices[0]][indices[1]]*S_inv[indices[0]][l]/S_inv[indices[0]][indices[0]]) +(S_inv[indices[1]][k]*(S_inv[indices[0]][l] - S_inv[indices[0]][indices[1]]*S_inv[indices[1]][l]/S_inv[indices[1]][indices[1]])))
        
        M = LA.inv(M_inv)
        
        return M

    # 以前に0とおいた偏相関係数が0から変化したかを確認する
    def _check(self, PCOR_mat):
        cnt = 0
        for idx in np.flipud(self.I):
            if abs(round(PCOR_mat[idx[0]-1][idx[1]-1],5)) >= 0.00001:
                return idx
            else:
                cnt += 1
        if cnt == len(self.I):
            return [-1, -1]

    # 反復して推定量を求める
    def check_ind(self, cnt, COV_mat, PCOR_mat, restore_flg=None):
        pcor_mat = PCOR_mat
        cov_mat = COV_mat
        cnt = 0

        while True:
            idx = self._check(pcor_mat)
            if idx[0] == -1:
                break

            cnt += 1
            idx = [i-1 for i in idx]
            # 制約条件下での分散共分散行列を推定
            est_COVmat = self.e_covmat(idx, cov_mat)
            # 制約条件下での相関行列を推定
            est_CORmat = self.cor_mat(est_COVmat)
            # 制約条件下での偏相関係数
            est_PCORmat = self.pcor_mat(est_CORmat)

            pcor_mat = est_PCORmat
            cov_mat = est_COVmat

        print('Number of iterations : ', cnt)
        print('\n')

        print('### after itteration ###')
        print('> estimated partial correlation matrix')
        pprint(pd.DataFrame(pcor_mat, index=self.rc_name, columns=[self.rc_name]))
        print('\n')

        print('> estimated correlation matrix')
        pprint(pd.DataFrame(self.cor_mat(cov_mat), index=self.rc_name, columns=[self.rc_name]))
        print('\n')

        # 計算結果を保存
        if restore_flg is not None:
          self.result['restore-{}'.format(restore_flg) + '- Number of iterations :{}'.format(cnt)] = cnt
          self.result['restore-{}'.format(restore_flg) + '-' + str(cnt+1) + ':pcor-mat(after)'] = pd.DataFrame(pcor_mat, index=self.rc_name, columns=[self.rc_name]).to_latex().replace('\n','')
          self.result['restore-{}'.format(restore_flg) + '-' + str(cnt+1) + ':cor-mat(after)'] = pd.DataFrame(self.cor_mat(cov_mat), index=self.rc_name, columns=[self.rc_name]).to_latex().replace('\n','')
        else:
          self.result['Number of iterations : {}'.format(cnt)] = cnt
          self.result[str(cnt+1) + ':pcor-mat(after)'] = pd.DataFrame(pcor_mat, index=self.rc_name, columns=[self.rc_name]).to_latex().replace('\n','')
          self.result[str(cnt+1) + ':cor-mat(after)'] = pd.DataFrame(self.cor_mat(cov_mat), index=self.rc_name, columns=[self.rc_name]).to_latex().replace('\n','')          
        return pcor_mat, self.cor_mat(cov_mat)

    # 共分散選択のメイン
    def do_covselection(self):
        cnt = 0
        restored_time = 0
        cov_mat = self.COV_mat
        while True:
            while True: # 削除する辺を入力
                indices_f = input('Enter index(from): ')
                if indices_f == 'End':
                    return True
                indices_t = input('Enter index(to): ')
                
                print('indices is ', [int(indices_f), int(indices_t)])
                ok_or_not = input('YES or NO : ')
                if ok_or_not == 'y':
                    break

            indices = [int(indices_f), int(indices_t)]
            print('indices: ', indices)
            indices = sorted(indices)

            self.result['comment-' + str(cnt+1)] = str(cnt+1)
            self.result['reduce idx-' + str(cnt+1)] = indices

            if len(self.I) == 0:
                self.I.append(indices)
            else:
                if indices not in self.I:
                    self.I.append(indices)
            indices = [int(indices_f)-1, int(indices_t)-1]

            # 制約条件下での分散共分散行列を推定
            est_COVmat = self.e_covmat(indices, cov_mat)
            # 制約条件下での相関行列を推定
            est_CORmat = self.cor_mat(est_COVmat)
            # 制約条件下での偏相関係数
            est_PCORmat = self.pcor_mat(est_CORmat)

            print('> estimated partial correlation matrix')
            pprint(pd.DataFrame(est_PCORmat, index=self.rc_name, columns=[self.rc_name]))
            print('\n')

            print('> estimated correlation matrix')
            pprint(pd.DataFrame(est_CORmat, index=self.rc_name, columns=[self.rc_name]))
            print('\n')

            self.result[str(cnt+1) + ':pcor-mat(before)'] = pd.DataFrame(est_PCORmat, index=self.rc_name, columns=[self.rc_name]).to_latex().replace('\n','')
            self.result[str(cnt+1) + ':cor-mat(before)'] = pd.DataFrame(est_CORmat, index=self.rc_name, columns=[self.rc_name]).to_latex().replace('\n','')

            if len(self.I) != 1:
              est_PCORmat, est_CORmat = self.check_ind(cnt, est_COVmat, est_PCORmat)
        
            print('I : ', self.I)
            print('\n')

            # 残差
            for idx in self.I:
                self.residual[idx[0]-1][idx[1]-1] = self.i_COR_mat[idx[0]-1][idx[1]-1] - est_CORmat[idx[0]-1][idx[1]-1]
            print('> residual (matrix)')
            pprint(pd.DataFrame(self.residual, index=self.rc_name, columns=[self.rc_name]))
            print('\n')

            # 逸脱度
            dev = self.data_np.shape[0] * math.log(LA.det(est_CORmat)/LA.det(self.COR_mat))

            cnt += 1
            p = chi2.sf(x = dev, df = len(self.I))
            print(str(cnt) + ' time')
            print('I : ', self.I)
            print('\n')

            print('deviance of model RM{}'.format(cnt))
            print('dev : ', '{:.3f}'.format(dev))
            print('degree of freedom : ', len(self.I))
            print('p : ', '{:.3f}'.format(p))
            print('\n')

            self.result['deviance of model RM{}'.format(cnt)] = '{:.3f}'.format(dev)
            self.result['RM{}-degree of freedom'.format(cnt)] = len(self.I)
            self.result['RM{}-p'.format(cnt)] = '{:.3f}'.format(p)
            
            # モデルの逸脱度の差
            if cnt >=2:
                print('difference between RM{}'.format(cnt) + ' and RM{}'.format(cnt-1))
                d_dev = dev - self.deviance[cnt-2]
                print('dev : ', '{:.3f}'.format(d_dev))
                print('degree of freedom : ', 1)
                d_p = chi2.sf(x = d_dev, df = 1)
                print('p : ', '{:.3f}'.format(d_p))
                print('\n')
                
                self.result['difference between RM{}'.format(cnt) + ' and RM{}'.format(cnt-1)] = '{:.3f}'.format(d_dev)
                self.result['(difference)RM{}-degree of freedom'.format(cnt)] = 1
                self.result['(difference)RM{}-p'.format(cnt)] = '{:.3f}'.format(d_p)

            y_n = input('want to restore the edges? : ') 
            if y_n == 'y': # 以前に削除した辺を復元する
                while True:
                    r_indices_f = input('Enter index(from): ')
                    r_indices_t = input('Enter index(to): ')
                
                    print('indices is ', [int(r_indices_f), int(r_indices_t)])
                    ok_or_not = input('YES or NO : ')
                    if ok_or_not == 'y':
                        break
                
                restored_time += 1
                r_indices = [int(r_indices_f), int(r_indices_t)]
                print('restore indices: ', r_indices)
                r_indices = sorted(r_indices)

                self.result['comment-' + str(restored_time)] = str(restored_time)
                self.result['restore idx -' + str(restored_time)] = r_indices

                tmp_I = []
                for idx in self.I:
                    if idx != r_indices:
                        tmp_I.append(idx)
                r_indices = [int(r_indices_f)-1, int(r_indices_t)-1]
                self.I = tmp_I
                self.residual = np.zeros((self.data_np.shape[1], self.data_np.shape[1]))
                cov_mat = self.i_COV_mat
                dev_tmp = []
                cnt = 0
                print('self.I', self.I)
                for idx in self.I:
                    print('remaked ; ' + str(cnt+1) + ' time')
                    self.result['restore-{}'.format(restored_time) + '-' +str(cnt+1) + ' time'] = str(cnt+1)

                    # 制約条件下での分散共分散行列を推定
                    t_idx = [idx[0]-1, idx[1]-1]
                    est_COVmat = self.e_covmat(t_idx, cov_mat)
                    # 制約条件下での相関行列を推定
                    est_CORmat = self.cor_mat(est_COVmat)
                    # 制約条件下での偏相関係数
                    est_PCORmat = self.pcor_mat(est_CORmat)

                    self.result['restore-{}'.format(restored_time) + '-' + str(cnt+1) + ':pcor-mat(before)'] = pd.DataFrame(est_PCORmat, index=self.rc_name, columns=[self.rc_name]).to_latex().replace('\n','')
                    self.result['restore-{}'.format(restored_time) + '-' + str(cnt+1) + ':cor-mat(before)'] = pd.DataFrame(est_CORmat, index=self.rc_name, columns=[self.rc_name]).to_latex().replace('\n','')

                    if cnt > 0:
                        est_PCORmat, est_CORmat = self.check_ind(cnt, est_COVmat, est_PCORmat, restore_flg=restored_time)

                    self.residual[idx[0]-1][idx[1]-1] = self.i_COR_mat[idx[0]-1][idx[1]-1] - est_CORmat[idx[0]-1][idx[1]-1]

                    # 逸脱度
                    dev = self.data_np.shape[0] * math.log(LA.det(est_CORmat)/LA.det(self.COR_mat))
                    dev_tmp.append(dev)
                    cnt += 1
                    p = chi2.sf(x = dev, df = cnt)
                    print('self.I : ', self.I)
                    print('idx : ', idx)
                    print('\n')
                    print('deviance of model RM{}'.format(cnt))
                    print('dev : ', '{:.3f}'.format(dev))
                    print('degree of freedom : ', cnt)
                    print('p : ', '{:.3f}'.format(p))
                    print('\n')

                    self.result['restore-{}'.format(restored_time) + '-deviance of model RM{}'.format(cnt)] = '{:.3f}'.format(dev)
                    self.result['restore-{}'.format(restored_time) + '-RM{}-degree of freedom'.format(cnt)] = cnt
                    self.result['restore-{}'.format(restored_time) + '-RM{}-p'.format(cnt)] = '{:.3f}'.format(p)

                    if cnt >= 2:
                        print('difference between RM{}'.format(cnt) + ' and RM{}'.format(cnt-1))
                        d_dev = dev_tmp[cnt-1] - dev_tmp[cnt-2]
                        print('dev : ', '{:.3f}'.format(d_dev))
                        print('degree of freedom : ', 1)
                        d_p = chi2.sf(x = d_dev, df = 1)
                        print('p : ', '{:.3f}'.format(d_p))
                        print('\n')
                        
                        self.result['restore-{}'.format(restored_time) + '-difference between RM{}'.format(cnt) + ' and RM{}'.format(cnt-1)] = '{:.3f}'.format(d_dev)
                        self.result['restore-{}'.format(restored_time) + '-(difference)RM{}-degree of freedom'.format(cnt)] = 1
                        self.result['restore-{}'.format(restored_time) + '-(difference)RM{}-p'.format(cnt)] = '{:.3f}'.format(d_p)

                    cov_mat = est_COVmat
                self.change_graph(cnt, r_indices, restore_flg=True)

            else: # 以前に削除した辺を復元しない
                self.deviance.append(dev)
                self.change_graph(cnt, indices)
                cov_mat = est_COVmat

def main():
    # 使用するデータのファイル名
    file_name = './csv_file_2019/men/final_men_points_2019-world.csv'

    data = pd.read_csv(file_name, header=0, index_col=0)
    CS = CovarianceSelection(data)
    CS.do_covselection() # 共分散選択を実行
    
    # 計算結果を保存するときのファイル名
    json_file_name = './result.json'
    # 計算結果をjsonファイルに保存する
    with open(json_file_name, "w") as f:
      json.dump(CS.result, f, indent=2)

if __name__ == '__main__':
    main()
