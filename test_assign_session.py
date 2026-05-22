import app
from flask import session

def test_assign_default_session(client):
    with client.session_transaction() as sess:
        sess['gender'] = 'Other'
        sess['age'] = '25'
        sess['preference'] = 'Men'
        sess['status'] = 'Single'
        sess['pref_o_sincere'] = '20'
        sess['pref_o_intelligence'] = '20'
        sess['pref_o_funny'] = '20'
        sess['pref_o_ambitious'] = '20'
        sess['pref_o_shared_interests'] = '20'
        for interest in app.INTEREST_FIELDS:
            sess[interest] = '5'
        sess['block1_set'] = 'A'
        sess['block2_set'] = 'B'
        sess['current_profile_idx'] = 0
        sess['ratings_block1'] = ['5','6','7']
        sess['ratings_block2'] = ['8','9','10']
        sess['id_block1'] = ['1','2','3']
        sess['id_block2'] = ['4','5','6']
        sess['train_acc'] = 0.85
        sess['agreement'] = 'yes'
        sess['trust'] = 'yes'
        sess['test_acc'] = 0.80
        sess['influence'] = 'medium'
        sess['comments'] = 'No comments.'
        sess['current_step'] = 9

def run_all_tests():
    app.app.config['TESTING'] = True
    with app.app.test_client() as client:
        test_assign_default_session(client)
        print('Session variables assigned for testing.')

if __name__ == '__main__':
    run_all_tests()
