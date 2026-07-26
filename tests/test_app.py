import unittest
import json
import base64
import numpy as np
import cv2
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, allowed_file


class EmotionAITestCase(unittest.TestCase):

    def setUp(self):
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        self.client = app.test_client()

    def test_allowed_file_extensions(self):
        """Test file extension validation helper"""
        self.assertTrue(allowed_file("test.jpg"))
        self.assertTrue(allowed_file("image.PNG"))
        self.assertTrue(allowed_file("photo.webp"))
        self.assertFalse(allowed_file("script.py"))
        self.assertFalse(allowed_file("document.pdf"))

    def test_home_page_get(self):
        """Test HTTP GET request to home page endpoint"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'EmotionAI', response.data)

    def test_live_page_get(self):
        """Test HTTP GET request to live emotion detection route"""
        response = self.client.get('/live')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Live', response.data)

    def test_about_and_contact_pages(self):
        """Test HTTP GET request to about and contact static pages"""
        res_about = self.client.get('/about')
        res_contact = self.client.get('/contact')
        self.assertEqual(res_about.status_code, 200)
        self.assertEqual(res_contact.status_code, 200)

    def test_security_headers_middleware(self):
        """Test that security headers are present on all responses"""
        response = self.client.get('/')
        self.assertEqual(response.headers.get('X-Content-Type-Options'), 'nosniff')
        self.assertEqual(response.headers.get('X-Frame-Options'), 'SAMEORIGIN')
        self.assertEqual(response.headers.get('X-XSS-Protection'), '1; mode=block')

    def test_api_detect_emotion_missing_payload(self):
        """Test API endpoint handles missing or invalid payload gracefully"""
        response = self.client.post('/api/detect_emotion', json={})
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertIn('error', data)

    def test_api_detect_emotion_synthetic_frame(self):
        """Test API endpoint with a valid synthetic image payload"""
        # Create a 100x100 synthetic test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', img)
        b64_str = base64.b64encode(buffer).decode('utf-8')
        payload = {"image": f"data:image/jpeg;base64,{b64_str}"}

        response = self.client.post('/api/detect_emotion', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('faces_count', data)
        self.assertIn('faces', data)


if __name__ == '__main__':
    unittest.main()
