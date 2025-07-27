#!/usr/bin/env python
"""
Check what's in our vector database for categories
"""
import asyncio
import logging
from qdrant_client import QdrantClient
import json

async def check_vector_contents():
    """Check what category vectors we have stored"""
    
    print("=== VECTOR DATABASE CONTENTS ===")
    
    # Connect to Qdrant
    client = QdrantClient(url="http://localhost:6333")
    
    # Get collection info
    collection_info = client.get_collection("itsm_categories")
    print(f"Collection points count: {collection_info.points_count}")
    
    # Get all points
    points = client.scroll(
        collection_name="itsm_categories",
        limit=100,
        with_payload=True,
        with_vectors=False
    )[0]
    
    print(f"\nFound {len(points)} category vectors:")
    
    target_categories = ["التسجيل", "الإرسالية", "المدفوعات", "تسجيل الدخول", "بيانات المنشأة"]
    
    for point in points:
        category_name = point.payload.get('name', 'Unknown')
        if category_name in target_categories:
            print(f"\n✅ {category_name}:")
            print(f"   Description: {point.payload.get('description', 'N/A')[:100]}...")
            print(f"   Keywords: {point.payload.get('keywords', [])}")
            print(f"   ID: {point.id}")

if __name__ == "__main__":
    asyncio.run(check_vector_contents())
